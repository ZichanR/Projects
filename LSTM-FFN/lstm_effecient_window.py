#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 19:36:52 2020

@author: zichan
"""

from numpy import array
from numpy import hstack, vstack
import numpy as np
import pandas as pd
import datetime
import pickle
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense, Dropout, Concatenate, Input
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

def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def softmax(x):
    orig_shape=x.shape
    if len(x.shape)>1:
        #Matrix
        #shift max whithin each row
        constant_shift=np.max(x,axis=1).reshape(1,-1)
        x-=constant_shift
        x=np.exp(x)
        normlize=np.sum(x,axis=1).reshape(1,-1)
        x/=normlize
    else:
        #vector
        constant_shift=np.max(x)
        x-=constant_shift
        x=np.exp(x)
        normlize=np.sum(x)
        x/=normlize
    assert x.shape==orig_shape
    return x

def plot_picture(history):
    '''
    画出训练集和验证集的损失和精度变化，分析模型状态
    :return:
    '''

    # 画出训练集和验证集的损失和精度变化，分析模型状态
    pre = history.history['precision']  # 训练集pre
    val_pre = history.history['val_precision']  # 验证集pre
    f1 = history.history['f1_score']
    val_f1 = history.history['val_f1_score']
    loss = history.history['loss']  # 训练损失
    val_loss = history.history['val_loss']  # 验证损失
    epochs = range(1, len(pre) + 1)  # 迭代次数
    plt.plot(epochs, loss, 'b--', label='Training loss')  # bo for blue dot 蓝色点
    plt.plot(epochs, val_loss, 'r--', label='Validation loss')
    plt.plot(epochs, pre, 'b+-', label='Training pre', markersize=4)  # bo for blue dot 蓝色点
    plt.plot(epochs, val_pre, 'r+-', label='Validation pre', markersize=4)
    plt.plot(epochs, f1, 'b.-', label='Training f1', markersize=4)  # bo for blue dot 蓝色点
    plt.plot(epochs, val_f1, 'r.-', label='Validation f1', markersize=4)
    plt.title('Training and validation loss, precision and f1-score')
    plt.xlabel('Epochs')
    plt.ylabel('Loss, precision and f1-score')
    plt.legend(loc='best',fontsize='x-small')
    plt.show()

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

#### n2v result
def load_n2v(n2vattribute_number):
    nodeemb_file = open('nodeembedding/allnode{}.nodeemb'.format(n2vattribute_number), 'r')
    fn2v = nodeemb_file.readlines()
    fn2v.pop(0)
    attributes = []
    for line in fn2v:
        attribute = []
        node1 = str(line.split(' ')[0].strip())
        attribute.append(node1)
        for i in range(1,(n2vattribute_number+1)):
            attribute1 = float(line.split(' ')[i].strip())
            attribute.append(attribute1)
        attributes.append(attribute)
    fn2vatt = np.matrix(attributes)
    return fn2vatt                            
        
        
def construct_input_data_all_timewindow(window, n_steps):
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
    x_train_list = np.array(x_train_list).reshape(-1, window*n_steps, n_features_raw)
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
    x_test_list = np.array(x_test_list).reshape(-1, window*n_steps, n_features_raw)
    out_all_list = np.array(out_all_list).reshape(-1, 1)
    test_mask_list = np.array(test_mask_list).reshape(-1,)
    return x_train_list, y_train_list, train_mask_list, x_test_list, out_all_list, test_mask_list

def construct_input_n2v(n2vattribute_number):
    fn2vatt = load_n2v(n2vattribute_number)
    n2v_input_train = []
    n2v_input_test = []
    n2v_input = []
    for eid in range(len(all_edge)):
        temp = hstack((fn2vatt[np.where(fn2vatt[:,0] == all_edge[eid][1])[0],1:(n2vattribute_number+1)], 
                       fn2vatt[np.where(fn2vatt[:,0] == all_edge[eid][2])[0],1:(n2vattribute_number+1)]))
        n2v_input.append(temp)
    n2v_input = np.array(n2v_input, dtype = np.float64).reshape(-1, n2vattribute_number*2)
    for tr_ind in range(len(train_date)):
        n2v_input_train.append(n2v_input)
    for te_ind in range(len(test_date)):
        n2v_input_test.append(n2v_input)
    n2v_input_train, n2v_input_test = np.array(n2v_input_train).reshape(-1, n2vattribute_number*2), np.array(n2v_input_test).reshape(-1, n2vattribute_number*2)
    return n2v_input_train, n2v_input_test
    
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

# ### nn_steps = [48, 24, 12]
# def construct_raw_features_new_step(nn_steps):
#     for date in all_date:
#         x_date = []
#         exec('x_date = x_{}'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
#         edges_fea = []
#         for e_id in range(x_date.shape[0]):
#             edge = x_date[e_id]
#             edge_vars = []
#             edge_fea = []
#             for nn in range(int(x_date.shape[1]/3)):
#                 edge_vars.append(np.sum([np.var(edge[nn*3:(nn*3+3),i]) for i in range(x_date.shape[2])]))
#             vars_order = pd.Series(edge_vars).rank(ascending=False, method='first')
#             n_select = int(x_date.shape[1]/3/(x_date.shape[1]/nn_steps))
#             for s in np.array(np.where(vars_order <= n_select)).reshape(-1).tolist():
#                 edge_fea.append(edge[s*3:(s*3+3),:])
#             edges_fea.append(np.array(edge_fea).reshape(-1, x_date.shape[2]))
#         edges_fea = np.array(edges_fea)
#         exec('x_{} = edges_fea'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
        
nn_steps = 48
for date in all_date:
    x_date = []
    exec('x_date = x_{}'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
    edges_fea = []
    for e_id in range(x_date.shape[0]):
        edge = x_date[e_id]
        edge_vars = []
        edge_fea = []
        for nn in range(int(x_date.shape[1]/3)):
            edge_vars.append(np.sum([np.var(edge[nn*3:(nn*3+3),i]) for i in range(x_date.shape[2])]))
        vars_order = pd.Series(edge_vars).rank(ascending=False, method='first')
        n_select = int(x_date.shape[1]/3/(x_date.shape[1]/nn_steps))
        for s in np.array(np.where(vars_order <= n_select)).reshape(-1).tolist():
            edge_fea.append(edge[s*3:(s*3+3),:])
        edges_fea.append(np.array(edge_fea).reshape(-1, x_date.shape[2]))
    edges_fea = np.array(edges_fea)
    exec('x_{} = edges_fea'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
n_steps = nn_steps

def run_LSTM_fit_all_timewindow(window, n_output = 1, n_epoch=100, classweight = [True, 'balanced'], n2v = [False, 0], activationfunc='sigmoid', checkpmonitor='loss'):
    ## get x_train_list, y_train_list, x_test_list, out_all_list
    x_train_all, y_train_all, train_mask_all, x_test_all, out_all_all, test_mask_all = construct_input_data_all_timewindow(window, n_steps)
    if n2v[0]:
        n2v_input_train, n2v_input_test = construct_input_n2v(n2v[1])
    
    # my_metrics = ['accuracy', metrics.Precision(name='precision'), metrics.Recall(name='recall')]
    my_metrics = ['accuracy', precision, recall, f1_score]
    reducelr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, 
                                 mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    # checkpoint = ModelCheckpoint(filepath=bst_weight_filepath, monitor='val_loss',mode='max' ,save_best_only='True')
    # callback_lists = [reducelr,checkpoint]
    # define alldata_model    
    
    raw_input = Input((n_steps * window,n_features))
    lstm_layer1 = LSTM(100, activation=activationfunc, return_sequences=True)(raw_input)
    lstm_layer2 = LSTM(100, activation=activationfunc)(lstm_layer1)
    dense_layer = Dense(n_features)(lstm_layer2)
    if n2v[0]:
        n2v_input = Input((n2v[1]*2,))
        concat_layer = Concatenate()([n2v_input, dense_layer])
        output = Dense(n_output)(concat_layer)
        model = Model(inputs = [raw_input,n2v_input], outputs = output)
    else:
        output = Dense(n_output)(dense_layer)
        model = Model(inputs = raw_input, outputs = output)
    
    # model_emb = Model(inputs = raw_input, outputs = dense_layer)
    
    # model = Sequential()
    # model.add(LSTM(100, activation=activationfunc, return_sequences=True, 
    #                input_shape=(n_steps * window, n_features)))#, kernel_initializer='he_normal'))
    # model.add(LSTM(100, activation=activationfunc))
    # #model.add(Dropout(0.2))
    # model.add(Dense(n_features))
    # model.add(Dense(n_output))
    
    if n_output == 1:
        ## For each example, there should be a single floating-point value per prediction.
        lstmloss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    else:
        ## There should be # classes floating point values per feature for y_pred and a single floating point value per feature for y_true.
        lstmloss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # define X, y. a single floating point value per feature for y_true
    X, y = x_train_all[np.where(train_mask_all==1)], y_train_all[np.where(train_mask_all==1)].astype(np.float32)
    
    if classweight[1] == None or classweight[1] == 'balanced':
        my_class_weight = compute_class_weight(classweight[1], np.unique(y), y.reshape(-1,)).tolist()
    else:
        my_class_weight = compute_class_weight('balanced', np.unique(y), y.reshape(-1,)).tolist()
        my_class_weight = [my_class_weight[0]*classweight[1], my_class_weight[1]/classweight[1]]
    class_weight_dict = dict(zip([x for x in [0,1]], my_class_weight))
    
    ### best model saved for each training set.
    bst_weight_filepath = 'model_outputloss/trainall_weights_window_{}_{}steps_output{}_{}_{}_{}_{}_n2v{}_{}.h5'.format(window, n_steps, n_output, classweight[0], classweight[1], n_epoch, activationfunc, n2v[1], checkpmonitor)
    checkpoint = ModelCheckpoint(filepath=bst_weight_filepath, monitor=checkpmonitor, mode='auto', save_best_only='True')
    # earlystop = EarlyStopping(monitor='loss', patience=30, verbose=0, mode='auto')
    
    print("\nTraining Modle. With {:d} day(s) time window, {:d} feature sets collected per day per edge, {:d} epochs and {} class weight during training, {:d} dimensions of node to vector input, {:s} as activation, {:d} final output dimensions, {:s} as checkpoint monitor.".format(window, n_steps, n_epoch, classweight[0], n2v[1], activationfunc, n_output, checkpmonitor))
    rmsprop = RMSprop(learning_rate=0.001)#, clipvalue=5.0)
    model.compile(optimizer=rmsprop, loss=lstmloss, metrics = my_metrics)
    
    
    # fit model
    if n2v[0]:
        history = model.fit([X, n2v_input_train[np.where(train_mask_all==1)]], y, epochs=n_epoch, validation_split=0.1, verbose=2, callbacks=[checkpoint,reducelr], class_weight=class_weight_dict)
    else:
        history = model.fit(X, y, epochs=n_epoch, validation_split=0.1, verbose=2, callbacks=[checkpoint,reducelr], class_weight=class_weight_dict)

    # if n2v[0]:
    #     history = model.fit([X, n2v_input_train[np.where(train_mask_all==1)]], y, epochs=n_epoch, validation_split=0.1, verbose=2, callbacks=[checkpoint,reducelr])
    # else:
    #     history = model.fit(X, y, epochs=n_epoch, validation_split=0.1, verbose=2, callbacks=[checkpoint,reducelr])
    
    historypath = 'HistoryDict/historydic_w_{}_{}steps_output{}_{}_{}_{}_{}_n2v{}_{}.txt'.format(window, n_steps, n_output, classweight[0], classweight[1], n_epoch, activationfunc, n2v[1], checkpmonitor)
    with open(historypath, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
            
    plot_picture(history)
    
    print('\nTraining Completed!')
    print('\n-------------------------')
    print('\nTesting Performance for model.\n')
    ### save performance

    performance_list = []
    performance_filepath = 'performance_time_window_outputloss/trainall_performance_w_{}_{}steps_output{}_{}_{}_{}_{}_n2v{}_{}.txt'.format(window, n_steps, n_output, classweight[0], classweight[1], n_epoch, activationfunc, n2v[1], checkpmonitor)
    model = load_model(bst_weight_filepath,
                       custom_objects={'precision': precision, 'recall': recall, 'f1_score':f1_score})
    x_test, y_out = x_test_all[np.where(test_mask_all==1)], out_all_all[np.where(test_mask_all==1)]
    if n2v[0]:
        yhat = model.predict([x_test, n2v_input_test[np.where(test_mask_all==1)]], verbose=1)
    else:
        yhat = model.predict(x_test, verbose=1)
    console_show = 'Prediction result: with {:d} day(s) time window, {:d} feature sets collected per day per edge, {:d} epochs and {} class weight during training, {:d} dimensions of node to vector input, {:s} as activation, {:d} final output dimensions, {:s} as checkpoint monitor.'.format(window, n_steps, n_epoch, classweight[0], n2v[1], activationfunc, n_output, checkpmonitor)
    print(console_show)
    console_show2 = '\nclass weight dictionary is {}'.format(class_weight_dict)
    print(console_show2)
    if n_output == 1:
        cla_rep = classification_report(y_out, [0 if sigmoid(i) < 0.5 else 1 for i in yhat])
        confu_matr = confusion_matrix(y_out, [0 if sigmoid(i) < 0.5 else 1 for i in yhat])
    else:
        cla_rep = classification_report(y_out, [np.argmax(softmax(i)) for i in yhat])
        confu_matr = confusion_matrix(y_out, [np.argmax(softmax(i)) for i in yhat])
    print(cla_rep)
    print(confu_matr)
    performance_list.append(console_show)
    performance_list.append(console_show2)
    performance_list.append(cla_rep)
    performance_list.append(confu_matr)
    performance_list.append(model.summary())
    file= open(performance_filepath, 'w')  
    for fp in performance_list:
        file.write(str(fp))
        file.write('\n')
    file.close()


# for n2vd in [8, 16, 64, 128]:
#     run_LSTM_fit_all_timewindow(window=1, n2v=[True, n2vd])
    
    
# run_LSTM_fit_all_timewindow(window=1, classweight=[True, 2])
# run_LSTM_fit_all_timewindow(window=1, classweight=[True, 3]) 
# run_LSTM_fit_all_timewindow(window=1, classweight=[True, 4]) 
# run_LSTM_fit_all_timewindow(window=1, classweight=[True, 1.5])

run_LSTM_fit_all_timewindow(window=1, n_epoch=200, classweight=[False, None], n2v=[True, 8])
run_LSTM_fit_all_timewindow(window=1, n_epoch=200, classweight=[False, None], n2v=[True, 128])