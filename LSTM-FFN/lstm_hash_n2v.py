#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 16:01:17 2020

@author: zichan
"""
from numpy import array
from numpy import hstack, vstack
import numpy as np
import pandas as pd
import datetime
import pickle
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
import sklearn.metrics as skmetrics

import warnings
# warnings.filterwarnings('ignore')

out_label = pd.read_csv('../data/pmads_data.csv')
# labelinfo = out_label.describe()
out_label = out_label[['Time','KEY','label']]

n2vattribute_number = 8

#### n2v result
def load_n2v():
    nodeemb_file = open('../all_data/allnode8.nodeemb', 'r')
    fn2v = nodeemb_file.readlines()
    fn2v.pop(0)
    node_num = 1854
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

train_date = ['2017/9/27','2017/9/28','2017/9/29','2017/9/30','2017/10/1','2017/10/2','2017/10/3',
              '2017/10/4','2017/10/5','2017/10/6','2017/10/7','2017/10/8','2017/10/9','2017/10/10',
              '2017/10/11','2017/10/12','2017/10/13','2017/10/14','2017/10/15','2017/10/16']
test_date = ['2017/10/17','2017/10/18','2017/10/19','2017/10/20','2017/10/21','2017/10/22']
all_date = ['2017/9/27','2017/9/28','2017/9/29','2017/9/30','2017/10/1','2017/10/2','2017/10/3',
              '2017/10/4','2017/10/5','2017/10/6','2017/10/7','2017/10/8','2017/10/9','2017/10/10',
              '2017/10/11','2017/10/12','2017/10/13','2017/10/14','2017/10/15','2017/10/16',
              '2017/10/17','2017/10/18','2017/10/19','2017/10/20','2017/10/21','2017/10/22']

# for date in all_date:
#     exec('x_{} = np.fromfile("../all_data/x_{}_3d_norm.bin")'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d'),datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
#     exec('x_{} = x_{}.reshape(-1,96,29)'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d'),datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))

fn2vatt = load_n2v()

### performance print-out
def evaluate_prediction(yhat, y_out):
    preds = []
    for i in range(len(yhat)):
        if yhat[i,0] > yhat[i,1]:
            preds.append([0])
        else:
            preds.append([1])
        labels = y_out
    roc_score = roc_auc_score(labels, preds)
    ap_score = average_precision_score(labels, preds)
    print(classification_report(labels, preds))
    print(confusion_matrix(labels, preds))
    return(roc_score,ap_score)

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

x_train = []
y_train = []
### merge data of 20 trainning days (X,y)/adding hash
for date in train_date:
    # exec('x_input=x_{}'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
    exec('x_input=np.fromfile("../all_data/x_{}_3d_norm.bin").reshape(-1,96,29)'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
    train_out_label = out_label[out_label['Time']==date]
    y = to_categorical(train_out_label['label'].values)
    keys = train_out_label['KEY'].values
    for i in range(len(x_input)):
        ## hash
        # temp = hstack((x_input[i],np.array([hash(keys[i][0:4])*(1e-19),hash(keys[i][12:16])*(1e-19)]*96).reshape(96,2)))
        ## n2v
        temp = hstack((x_input[i],np.array([fn2vatt[np.where(fn2vatt[:,0] == keys[i][0:4])[0],1:(n2vattribute_number+1)], fn2vatt[np.where(fn2vatt[:,0] == keys[i][12:16])[0],1:(n2vattribute_number+1)]]*96).reshape(96,-1)))
        x_train.append(temp)
        y_train.append(y[i])
x_train = array(x_train, dtype = np.float64)
y_train = array(y_train)

x_test = []
out_all = []
### merge data of 6 testing days (X,y)
for date in test_date:
    # exec('x_input=x_{}'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
    exec('x_input=np.fromfile("../all_data/x_{}_3d_norm.bin").reshape(-1,96,29)'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
    test_out_label = out_label[out_label['Time']==date]
    y_out = test_out_label['label'].values
    out_all.append(y_out)
    keys = test_out_label['KEY'].values
    for i in range(len(x_input)):
        ## hash
        # temp = hstack((x_input[i],np.array([hash(keys[i][0:4])*(1e-19),hash(keys[i][12:16])*(1e-19)]*96).reshape(96,2)))
        ## n2v
        temp = hstack((x_input[i],np.array([fn2vatt[np.where(fn2vatt[:,0] == keys[i][0:4])[0],1:(n2vattribute_number+1)], fn2vatt[np.where(fn2vatt[:,0] == keys[i][12:16])[0],1:(n2vattribute_number+1)]]*96).reshape(96,-1)))
        x_test.append(temp)
x_test = array(x_test, dtype = np.float64)
out_all = array(out_all)
out_all = array([i for j in out_all for i in j])

alltrain_reducelr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
alltrain_checkpoint = ModelCheckpoint(filepath='model/alltrain_n2v_nodense_weights.h5',monitor='loss',mode='auto' ,save_best_only='True')
alltrain_callback_lists = [alltrain_reducelr,alltrain_checkpoint]

n_steps = 96
n_features = x_train.shape[2]

my_class_weight = compute_class_weight('balanced',np.unique(y_train), y_train.reshape(-1,)).tolist()
class_weight_dict = dict(zip([x for x in [0,1]], my_class_weight))

my_metrics = ['accuracy', precision, recall, f1_score]

# define alldata_model
alldata_model = Sequential()
alldata_model.add(LSTM(100, activation='sigmoid', return_sequences=True, input_shape=(n_steps, n_features), kernel_initializer='he_normal'))
alldata_model.add(LSTM(100, activation='sigmoid'))
#model.add(Dropout(0.2))
# alldata_model.add(Dense(n_features))
alldata_model.add(Dense(2,activation='softmax'))

rmsprop = RMSprop(learning_rate=0.001)
alldata_model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics = my_metrics)
alldata_model.fit(x_train, y_train, epochs=100, verbose=2, 
                  validation_split=0.1, callbacks = alltrain_callback_lists, class_weight=class_weight_dict)

alldata_model = load_model('model/alltrain_n2v_nodense_weights.h5',custom_objects={'precision': precision, 'recall': recall, 'f1_score':f1_score})

yhat_all = alldata_model.predict(x_test, verbose=1)
print(classification_report(out_all, [np.argmax(i) for i in yhat_all]))
print(confusion_matrix(out_all, [np.argmax(i) for i in yhat_all]))

# evaluate_prediction(yhat_all, out_all)

