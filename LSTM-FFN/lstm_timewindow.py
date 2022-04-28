#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 16:00:38 2020

@author: zichan
"""
from numpy import array
from numpy import hstack, vstack
import numpy as np
import pandas as pd
import datetime
import pickle
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, multilabel_confusion_matrix

import warnings
warnings.filterwarnings('ignore')

### performance print-out
def evaluate_prediction(yhat, y_out):
    preds = []
    for i in range(len(yhat)):
        if yhat[i,0] > yhat[i,1]:
            preds.append([0])
        else:
            preds.append([1])
        labels = y_out
    print(classification_report(labels, preds))
    print(confusion_matrix(labels, preds))
    return

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    true_negatives = K.sum(K.round(K.clip( (1-y_true) * (1-y_pred), 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    predicted_negatives = K.sum(1-K.round(K.clip(y_pred, 0, 1)))
    precision_positives = true_positives / (predicted_positives + K.epsilon())
    precision_negatives = true_negatives / (predicted_negatives + K.epsilon())
    mac_precision = (precision_positives + precision_negatives)/2
    return mac_precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    true_negatives = K.sum(K.round(K.clip( (1-y_true) * (1-y_pred), 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    possible_negatives = K.sum(1-K.round(K.clip(y_true, 0, 1)))
    recall_positives = true_positives / (possible_positives + K.epsilon())
    recall_negatives = true_negatives / (possible_negatives + K.epsilon())
    mac_recall = (recall_positives + recall_negatives)/2
    return mac_recall

def f1_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    # if K.sum(K.round(K.clip(y_true, 0, 1))) == tf.constant(0.0, dtype=(tf.float32)):
    #     return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def construct_input_data_list_timewindow(window):
    ### getting lists of training data and labels within time window
    x_train_list = []
    y_train_list = []
    train_mask_list = [] ### train the model with existing edges
    for tr_ind in range(len(train_date)):
        date = train_date[tr_ind]
        train_element = []
        for key_ind in range(len(all_edge)):
            edge_ele = np.zeros([n_steps * window, n_features_raw]) ## shape(96*window, 29)
            for w in range(window):
                if (w - window + 1 + tr_ind) >= 0 :
                    d = all_date[w - window + 1 + tr_ind]
                    exec('edge_ele[n_steps*w:n_steps*(w+1),:] = x_{}[key_ind,:,:].reshape(n_steps,n_features_raw)'.format(datetime.datetime.strptime(d,'%Y/%m/%d').date().strftime('%m%d'))) ## shape(96, 29)
                    # edge_ele[n_steps*w:n_steps*(w+1),:] = edge_date_ele
            train_element.append(edge_ele)
        x_train_list.append(np.array(train_element))
        
        ### label of a train_element is the label of edges from last day
        train_out_label = out_label[out_label['Time']==date]
        y_orig = train_out_label['label'].values # shape(n_existingedges,)
        ## expand labels assign [1,0] as label to non-existing edges
        x_input_keys = train_out_label['KEY'].values
        y = []
        for key_ind in range(len(all_edge)):
            if all_edge[key_ind,0] in x_input_keys:
                y.append(y_orig[np.where(x_input_keys==all_edge[key_ind,0])])
            else:
                y.append(np.array([-1]))
        y = np.array(y).reshape(-1,1)
        y_train_list.append(y)
        train_mask = np.array([0 if i == -1 else 1 for i in y])
        train_mask_list.append(train_mask)

    x_test_list = []
    out_all_list = []
    test_mask_list = [] ### test the model with existing edges
    for te_ind in range(len(test_date)):
        date = test_date[te_ind]
        test_element = []
        for key_ind in range(len(all_edge)):
            edge_ele = np.zeros([n_steps * window, n_features_raw]) ## shape(96*window, 29)
            for w in range(window):
                ## part of training data could be test data 
                ## (i.e. window = 2, x_test_list[0] is the last day and first day of edge data combined)
                if (w - window + 1 + te_ind + len(train_date)) >= 0 :
                    d = all_date[w - window + 1 + te_ind + len(train_date)]
                    exec('edge_ele[n_steps*w:n_steps*(w+1),:] = x_{}[key_ind,:,:].reshape(n_steps,n_features_raw)'.format(datetime.datetime.strptime(d,'%Y/%m/%d').date().strftime('%m%d'))) ## shape(96, 29)
                    # edge_ele[n_steps*w:n_steps*(w+1),:] = edge_date_ele
            test_element.append(edge_ele)
        x_test_list.append(np.array(test_element))
        
        ### groundtruth of a test_element is the labels of edges from last day shape(2142,1)
        test_out_label = out_label[out_label['Time']==date]
        y_out = test_out_label['label'].values.reshape(-1,1) # shape(n_existingedges,1)
        ## expand labels assign 0 as label to non-existing edges
        x_input_keys = test_out_label['KEY'].values
        out_all = []
        for key_ind in range(len(all_edge)):
            if all_edge[key_ind,0] in x_input_keys:
                out_all.append(y_out[np.where(x_input_keys==all_edge[key_ind,0])])
            else:
                out_all.append(np.array([-1]).reshape(-1,1))
        out_all = np.array(out_all).reshape(-1,1)
        out_all_list.append(out_all)
        test_mask = np.array([0 if i == -1 else 1 for i in out_all])
        test_mask_list.append(test_mask)
    return x_train_list, y_train_list, train_mask_list, x_test_list, out_all_list, test_mask_list

def construct_input_data_all_timewindow(window):
    ### getting lists of training data and labels within time window
    x_train_list = []
    y_train_list = []
    train_mask_list = [] ### train the model with existing edges
    for tr_ind in range(len(train_date)):
        date = train_date[tr_ind]
        train_element = []
        for key_ind in range(len(all_edge)):
            edge_ele = np.zeros([n_steps * window, n_features_raw]) ## shape(96*window, 29)
            for w in range(window):
                if (w - window + 1 + tr_ind) >= 0 :
                    d = all_date[w - window + 1 + tr_ind]
                    exec('edge_ele[n_steps*w:n_steps*(w+1),:] = x_{}[key_ind,:,:].reshape(n_steps,n_features_raw)'.format(datetime.datetime.strptime(d,'%Y/%m/%d').date().strftime('%m%d'))) ## shape(96, 29)
                    # edge_ele[n_steps*w:n_steps*(w+1),:] = edge_date_ele
            train_element.append(edge_ele)
        x_train_list.append(np.array(train_element))
        
        ### label of a train_element is the label of edges from last day
        train_out_label = out_label[out_label['Time']==date]
        y_orig = train_out_label['label'].values # shape(n_existingedges,)
        ## expand labels assign [1,0] as label to non-existing edges
        x_input_keys = train_out_label['KEY'].values
        y = []
        for key_ind in range(len(all_edge)):
            if all_edge[key_ind,0] in x_input_keys:
                y.append(y_orig[np.where(x_input_keys==all_edge[key_ind,0])])
            else:
                y.append(np.array([-1]))
        y = np.array(y).reshape(-1,1)
        y_train_list.append(y)
        train_mask = np.array([0 if i == -1 else 1 for i in y])
        train_mask_list.append(train_mask)
    x_train_list = np.array(x_train_list).reshape(-1, window*96, 29)
    y_train_list = np.array(y_train_list).reshape(-1, 1)
    train_mask_list = np.array(train_mask_list).reshape(-1,)
    
    x_test_list = []
    out_all_list = []
    test_mask_list = [] ### test the model with existing edges
    for te_ind in range(len(test_date)):
        date = test_date[te_ind]
        test_element = []
        for key_ind in range(len(all_edge)):
            edge_ele = np.zeros([n_steps * window, n_features_raw]) ## shape(96*window, 29)
            for w in range(window):
                ## part of training data could be test data 
                ## (i.e. window = 2, x_test_list[0] is the last day and first day of edge data combined)
                if (w - window + 1 + te_ind + len(train_date)) >= 0 :
                    d = all_date[w - window + 1 + te_ind + len(train_date)]
                    exec('edge_ele[n_steps*w:n_steps*(w+1),:] = x_{}[key_ind,:,:].reshape(n_steps,n_features_raw)'.format(datetime.datetime.strptime(d,'%Y/%m/%d').date().strftime('%m%d'))) ## shape(96, 29)
                    # edge_ele[n_steps*w:n_steps*(w+1),:] = edge_date_ele
            test_element.append(edge_ele)
        x_test_list.append(np.array(test_element))
        
        ### groundtruth of a test_element is the labels of edges from last day shape(2142,1)
        test_out_label = out_label[out_label['Time']==date]
        y_out = test_out_label['label'].values.reshape(-1,1) # shape(n_existingedges,1)
        ## expand labels assign 0 as label to non-existing edges
        x_input_keys = test_out_label['KEY'].values
        out_all = []
        for key_ind in range(len(all_edge)):
            if all_edge[key_ind,0] in x_input_keys:
                out_all.append(y_out[np.where(x_input_keys==all_edge[key_ind,0])])
            else:
                out_all.append(np.array([-1]).reshape(-1,1))
        out_all = np.array(out_all).reshape(-1,1)
        out_all_list.append(out_all)
        test_mask = np.array([0 if i == -1 else 1 for i in out_all])
        test_mask_list.append(test_mask)
    x_test_list = np.array(x_test_list).reshape(-1, window*96, 29)
    out_all_list = np.array(out_all_list).reshape(-1, 1)
    test_mask_list = np.array(test_mask_list).reshape(-1,)
    return x_train_list, y_train_list, train_mask_list, x_test_list, out_all_list, test_mask_list


out_label = pd.read_csv('../data/pmads_data.csv')
# labelinfo = out_label.describe()
out_label = out_label[['Time','KEY','label']]

all_edge = np.load("../all_data/static_all_edge.edge.npy", allow_pickle=True)

n_steps = 96
n_features_raw = 29
n_edges = len(all_edge)
n_features = n_features_raw

train_date = ['2017/9/27','2017/9/28','2017/9/29','2017/9/30','2017/10/1','2017/10/2','2017/10/3',
              '2017/10/4','2017/10/5','2017/10/6','2017/10/7','2017/10/8','2017/10/9','2017/10/10',
              '2017/10/11','2017/10/12','2017/10/13','2017/10/14','2017/10/15','2017/10/16']
test_date = ['2017/10/17','2017/10/18','2017/10/19','2017/10/20','2017/10/21','2017/10/22']
all_date = ['2017/9/27','2017/9/28','2017/9/29','2017/9/30','2017/10/1','2017/10/2','2017/10/3',
              '2017/10/4','2017/10/5','2017/10/6','2017/10/7','2017/10/8','2017/10/9','2017/10/10',
              '2017/10/11','2017/10/12','2017/10/13','2017/10/14','2017/10/15','2017/10/16',
              '2017/10/17','2017/10/18','2017/10/19','2017/10/20','2017/10/21','2017/10/22']

## get preprocessed 3d norm all edge 
for date in all_date:
    exec('x_{} = np.fromfile("../all_data/staticedge_date_3d_norm/x_{}_3d_norm_ae.bin").reshape(-1,n_steps,n_features_raw)'
         .format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d'),datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))


def run_LSTM_fit_day_timewindow(window):
    modelweightspath = 'model/model_weights_w_{}'.format(window)
    if not os.path.exists(modelweightspath):
        os.makedirs(modelweightspath)
    ## get x_train_list, y_train_list, x_test_list, out_all_list
    x_train_list, y_train_list, train_mask_list, x_test_list, out_all_list, test_mask_list = construct_input_data_list_timewindow(window)
    
    # my_metrics = ['accuracy', metrics.Precision(name='precision'), metrics.Recall(name='recall')]
    my_metrics = ['accuracy', precision, recall, f1_score]
    reducelr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    # checkpoint = ModelCheckpoint(filepath=bst_weight_filepath, monitor='val_loss',mode='max' ,save_best_only='True')
    # callback_lists = [reducelr,checkpoint]
    # define alldata_model
    model = Sequential()
    model.add(LSTM(100, activation='sigmoid', return_sequences=True, 
                   input_shape=(n_steps * window, n_features), kernel_initializer='he_normal'))
    model.add(LSTM(100, activation='sigmoid'))
    #model.add(Dropout(0.2))
    model.add(Dense(n_features))
    # model.add(Dense(2,activation='softmax'))
    model.add(Dense(1,activation='sigmoid'))
    
    ## fit train_list in model
    for tr_ind in range(len(train_date)):
    # for tr_ind in [18,19]:
        x_train = x_train_list[tr_ind]
        y_train = y_train_list[tr_ind]
        train_mask = train_mask_list[tr_ind]
        date = train_date[tr_ind]
        X, y = x_train[np.where(train_mask==1)], y_train[np.where(train_mask==1)]
        ### best model saved for each training set.
        bst_weight_filepath = 'model/model_weights_w_{}/weights_window_{}_traindate_{}.h5'.format(window,window,datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d'))
        checkpoint = ModelCheckpoint(filepath=bst_weight_filepath, monitor='val_f1_score',mode='max' ,save_best_only='True')
        earlystop = EarlyStopping(monitor='loss', patience=30, verbose=0, mode='auto')
        # monitor是需要监视的常量, patience是当early stop被激活之后, 再尝试进行多少次训练.

        my_class_weight = compute_class_weight('balanced',np.unique(y), y.reshape(-1,)).tolist()
        class_weight_dict = dict(zip([x for x in [0,1]], my_class_weight))
        # fit model
        print("\nTraining Modle. Date: ","{:s}.".format(date),
              "With ", "{:d} days time window.".format(window))
        if tr_ind > 0 :
            previous_bst_weight_filepath = 'model/model_weights_w_{}/weights_window_{}_traindate_{}.h5'.format(window,window,datetime.datetime.strptime(train_date[tr_ind-1],'%Y/%m/%d').date().strftime('%m%d'))
            model = load_model(previous_bst_weight_filepath, custom_objects={'precision': precision, 'recall': recall, 'f1_score':f1_score})
            rmsprop = RMSprop(learning_rate=0.001)#, clipvalue=5.0)
            model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics = my_metrics)
            history = model.fit(X, y, epochs=200, validation_split=0.1, verbose=2, callbacks=[checkpoint,reducelr, earlystop], class_weight=class_weight_dict)
        else:
            rmsprop = RMSprop(learning_rate=0.001)#, clipvalue=5.0)
            model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics = my_metrics)
            history = model.fit(X, y, epochs=200, validation_split=0.1, verbose=2, callbacks=checkpoint, class_weight=class_weight_dict)
        with open('trainwCallbackHistoryDict.txt', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    print('\nTraining Completed!')
    print('\n-------------------------')
    print('\nTesting Performance for model.')
    ### save performance
    performance_list = []
    performance_filepath = 'performance_time_window/performance_w_{}.txt'.format(window)
    model = load_model('model/model_weights_w_{}/weights_window_{}_traindate_{}.h5'.format(window,window,datetime.datetime.strptime(train_date[len(train_date)-1],'%Y/%m/%d').date().strftime('%m%d')), 
                       custom_objects={'precision': precision, 'recall': recall, 'f1_score':f1_score})
    for te_ind in range(len(test_date)):
        x_test = x_test_list[te_ind]
        y_out = out_all_list[te_ind]
        test_mask = test_mask_list[te_ind]
        date = test_date[te_ind]
        yhat = model.predict(x_test[np.where(test_mask==1)], verbose=1)
        console_show = 'Predictions for date: {:s}. With {:d} days time window.'.format(date,window)
        print(console_show)
        cla_rep = classification_report(y_out[np.where(test_mask==1)], [0 if i < 0.5 else 1 for i in yhat])
        confu_matr = confusion_matrix(y_out[np.where(test_mask==1)], [0 if i < 0.5 else 1 for i in yhat])
        print(cla_rep)
        performance_list.append(console_show)
        performance_list.append(cla_rep)
        performance_list.append(confu_matr)
        performance_list.append('\n')
    
    file= open(performance_filepath, 'w')  
    for fp in performance_list:
        file.write(str(fp))
        file.write('\n')
    file.close()

def run_LSTM_fit_all_timewindow(window, n_output, n_epoch=100, classweight = True, activationfunc='sigmoid', checkpmonitor='val_f1_score'):
    ## get x_train_list, y_train_list, x_test_list, out_all_list
    x_train_all, y_train_all, train_mask_all, x_test_all, out_all_all, test_mask_all = construct_input_data_all_timewindow(window)
    
    # my_metrics = ['accuracy', metrics.Precision(name='precision'), metrics.Recall(name='recall')]
    my_metrics = ['accuracy', precision, recall, f1_score]
    reducelr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=20, verbose=1, 
                                 mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    # checkpoint = ModelCheckpoint(filepath=bst_weight_filepath, monitor='val_loss',mode='max' ,save_best_only='True')
    # callback_lists = [reducelr,checkpoint]
    # define alldata_model
    model = Sequential()
    model.add(LSTM(100, activation=activationfunc, return_sequences=True, 
                   input_shape=(n_steps * window, n_features), kernel_initializer='he_normal'))
    model.add(LSTM(100, activation=activationfunc))
    #model.add(Dropout(0.2))
    model.add(Dense(n_features))
    if n_output == 1:
        model.add(Dense(n_output), 'sigmoid')
        X, y = x_train_all[np.where(train_mask_all==1)], y_train_all[np.where(train_mask_all==1)]
        lstmloss = tf.keras.losses.BinaryCrossentropy()
    else:
        model.add(Dense(n_output), 'softmax')
        X, y = x_train_all[np.where(train_mask_all==1)], to_categorical(y_train_all[np.where(train_mask_all==1)])
        lstmloss = tf.keras.losses.BinaryCrossentropy()
        # X, y = x_train_all[np.where(train_mask_all==1)], y_train_all[np.where(train_mask_all==1)]
        # lstmloss = tf.keras.losses.SparseCategoricalCrossentropy()    
    
    ### best model saved for each training set.
    bst_weight_filepath = 'model/trainall_weights_window_{}_output{}_{}_{}_{}_{}.h5'.format(window, n_output, classweight, n_epoch, activationfunc, checkpmonitor)
    checkpoint = ModelCheckpoint(filepath=bst_weight_filepath, monitor=checkpmonitor,mode='auto' ,save_best_only='True')
    # earlystop = EarlyStopping(monitor='loss', patience=30, verbose=0, mode='auto')
    
    if classweight:
        my_class_weight = compute_class_weight('balanced',np.unique(y), y.reshape(-1,)).tolist()
        class_weight_dict = dict(zip([x for x in [0,1]], my_class_weight))
        # fit model
        print("\nTraining Modle. With {:d} day(s) time window, {:d} epochs and {} class weight during training, {:s} as activation, {:d} final output dimensions, {:s} as checkpoint monitor.".format(window, n_epoch, classweight, activationfunc, n_output, checkpmonitor))
        rmsprop = RMSprop(learning_rate=0.001)#, clipvalue=5.0)
        model.compile(optimizer=rmsprop, loss=lstmloss, metrics = my_metrics)
        history = model.fit(X, y, epochs=n_epoch, validation_split=0.1, verbose=2, callbacks=[checkpoint,reducelr], class_weight=class_weight_dict)
        with open('trainwCallbackHistoryDict.txt', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    else:
        print("\nTraining Modle. With {:d} day(s) time window, {:d} epochs and {} class weight during training, {:s} as activation, {:d} final output dimensions, {:s} as checkpoint monitor.".format(window, n_epoch, classweight, activationfunc, n_output, checkpmonitor))
        rmsprop = RMSprop(learning_rate=0.001)#, clipvalue=5.0)
        model.compile(optimizer=rmsprop, loss=lstmloss, metrics = my_metrics)
        history = model.fit(X, y, epochs=n_epoch, validation_split=0.1, verbose=2, callbacks=[checkpoint,reducelr])
        with open('trainwCallbackHistoryDict.txt', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
            
    print('\nTraining Completed!')
    print('\n-------------------------')
    print('\nTesting Performance for model.\n')
    ### save performance

    performance_list = []
    performance_filepath = 'performance_time_window/trainall_performance_w_{}_output{}_{}_{}_{}_{}.txt'.format(window,n_output, classweight, n_epoch, activationfunc, checkpmonitor)
    model = load_model('model/trainall_weights_window_{}_output{}_{}_{}_{}_{}.h5'.format(window, n_output, classweight, n_epoch, activationfunc, checkpmonitor),
                        custom_objects={'precision': precision, 'recall': recall, 'f1_score':f1_score})
    x_test, y_out = x_test_all[np.where(test_mask_all==1)], out_all_all[np.where(test_mask_all==1)]
    yhat = model.predict(x_test, verbose=1)
    console_show = 'Prediction result: with {:d} day(s) time window, {:d} epochs and {} class weight during training, {:s} as activation, {:d} final output dimensions, {:s} as checkpoint monitor.'.format(window, n_epoch, classweight, activationfunc, n_output, checkpmonitor)
    print(console_show)
    if n_output == 1:
        cla_rep = classification_report(y_out, [0 if i < 0.5 else 1 for i in yhat])
        confu_matr = confusion_matrix(y_out, [0 if i < 0.5 else 1 for i in yhat])
    else:
        cla_rep = classification_report(y_out, [np.argmax(i) for i in yhat])
        confu_matr = confusion_matrix(y_out, [np.argmax(i) for i in yhat])
    print(cla_rep)
    print(confu_matr)
    performance_list.append(console_show)
    performance_list.append(cla_rep)
    performance_list.append(confu_matr)
    file= open(performance_filepath, 'w')  
    for fp in performance_list:
        file.write(str(fp))
        file.write('\n')
    file.close()
    
# run_LSTM_fit_all_timewindow(window=2, n_output=2, n_epoch=100, classweight=True, activationfunc='sigmoid', checkpmonitor='loss')
# run_LSTM_fit_all_timewindow(window=2, n_output=2, n_epoch=100, classweight=False, activationfunc='sigmoid', checkpmonitor='loss')
# run_LSTM_fit_all_timewindow(window=3, n_output=2, n_epoch=100, classweight=True, activationfunc='sigmoid', checkpmonitor='loss')
# run_LSTM_fit_all_timewindow(window=3, n_output=2, n_epoch=100, classweight=False, activationfunc='sigmoid', checkpmonitor='loss')
