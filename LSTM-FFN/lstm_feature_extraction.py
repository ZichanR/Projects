#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 18:14:48 2020

@author: zichan
"""

# multivariate data preparation
from numpy import array
from numpy import hstack
import numpy as np
import pandas as pd
import datetime
import pickle
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

in_rawpmads = pd.read_csv('../all_data/all_data.csv')
pmadsinfo = in_rawpmads.describe()


out_label = pd.read_csv('../data/pmads_data.csv')
labelinfo = out_label.describe()
out_label = out_label[['Time','KEY','label']]

#### sanitize raw input data: in_rawpmads.
## split 'Time' into 'Date' and 'Time'
col_names = in_rawpmads.columns.tolist()
col_names.insert(1,'Date') # add a new column represent date
in_rawpmads = in_rawpmads.reindex(columns=col_names)
dt = array([in_rawpmads.loc[i,'Time'].split(' ', 1 ) for i in range(len(in_rawpmads))])
in_rawpmads['Date'] = dt[:,0]
in_rawpmads['Time'] = dt[:,1]

time_name = array(['00:00:00', '00:15:00', '00:30:00', '00:45:00', '01:00:00',
       '01:15:00', '01:30:00', '01:45:00', '02:00:00', '02:15:00',
       '02:30:00', '02:45:00', '03:00:00', '03:15:00', '03:30:00',
       '03:45:00', '04:00:00', '04:15:00', '04:30:00', '04:45:00',
       '05:00:00', '05:15:00', '05:30:00', '05:45:00', '06:00:00',
       '06:15:00', '06:30:00', '06:45:00', '07:00:00', '07:15:00',
       '07:30:00', '07:45:00', '08:00:00', '08:15:00', '08:30:00',
       '08:45:00', '09:00:00', '09:15:00', '09:30:00', '09:45:00',
       '10:00:00', '10:15:00', '10:30:00', '10:45:00', '11:00:00',
       '11:15:00', '11:30:00', '11:45:00', '12:00:00', '12:15:00',
       '12:30:00', '12:45:00', '13:00:00', '13:15:00', '13:30:00',
       '13:45:00', '14:00:00', '14:15:00', '14:30:00', '14:45:00',
       '15:00:00', '15:15:00', '15:30:00', '15:45:00', '16:00:00',
       '16:15:00', '16:30:00', '16:45:00', '17:00:00', '17:15:00',
       '17:30:00', '17:45:00', '18:00:00', '18:15:00', '18:30:00',
       '18:45:00', '19:00:00', '19:15:00', '19:30:00', '19:45:00',
       '20:00:00', '20:15:00', '20:30:00', '20:45:00', '21:00:00',
       '21:15:00', '21:30:00', '21:45:00', '22:00:00', '22:15:00',
       '22:30:00', '22:45:00', '23:00:00', '23:15:00', '23:30:00',
       '23:45:00'])

#train_out_label = out_label.loc[0:100,:]
train_date = ['2017/9/27','2017/9/28','2017/9/29','2017/9/30','2017/10/1','2017/10/2','2017/10/3',
              '2017/10/4','2017/10/5','2017/10/6','2017/10/7','2017/10/8','2017/10/9','2017/10/10',
              '2017/10/11','2017/10/12','2017/10/13','2017/10/14','2017/10/15','2017/10/16',
              '2017/10/17']
test_date = ['2017/10/18','2017/10/19','2017/10/20','2017/10/21','2017/10/22']
all_date = ['2017/9/27','2017/9/28','2017/9/29','2017/9/30','2017/10/1','2017/10/2','2017/10/3',
              '2017/10/4','2017/10/5','2017/10/6','2017/10/7','2017/10/8','2017/10/9','2017/10/10',
              '2017/10/11','2017/10/12','2017/10/13','2017/10/14','2017/10/15','2017/10/16',
              '2017/10/17','2017/10/18','2017/10/19','2017/10/20','2017/10/21','2017/10/22']

# reconstruct inputs: key-date-label as a sequence. 
# fill-up 96-slots a day for all links exist in out_label
# a batch of sequences constructed as a input 3D(batch_size, time_steps, features_num).
def reconstruct_lstm_input(label):
    temp_input = list()
    ## get [Key, Date] set entity
    # check if two date are identical
    # datetime.datetime.strptime(out_label[0,0],'%Y/%m/%d').date() == datetime.datetime.strptime('2017-09-27','%Y-%m-%d').date()
    test_input_rawdata = [in_rawpmads[(in_rawpmads['KEY']==label.iloc[i,1]) 
                                        & (in_rawpmads['Date']==str(datetime.datetime.strptime(label.iloc[i,0],'%Y/%m/%d').date()))]
                      for i in range(len(label))]
    
    ## [Key, Date] entity
    for entity in test_input_rawdata:
        ## check for duplication
        freq = entity['Time'].value_counts()
        dup_time = freq[freq>1].index.tolist()
        ## generate new optimized features from duplicated timestamps, delete old ones
        if len(dup_time):
            for dup in dup_time:
                duplications = entity[entity['Time']==dup]
                dup_ind = entity[entity['Time']==dup].index.tolist()
                entity = entity.drop(index=dup_ind)
                entity = entity.append(duplications.max(),ignore_index=True)
                
        ## less than 96 unique timestamps
        if len(entity['Time'].unique()) != 96:
            for time in time_name:
                # check if current timestamp exists
                if time not in entity['Time'].values:
                    entity = entity.append([{'Time':str(time), 'Date':entity['Date'].max(), 'KEY':entity['KEY'].max()}], ignore_index=True)
        
        ## sort by time
        entity.sort_values(by='Time')
        
        ## add nan/zero info sub_column for 8 columns 
        # -1 for nans / 0 for 0s / 1 for nonzeros    
        # itcol_name = ['Source_SNR_MAX', 'Sink_SNR_MAX', 'Source_SNR_MIN','Sink_SNR_MIN', 
        #               'Source_RSL_MAX', 'Sink_RSL_MAX', 'Source_RSL_MIN', 'Sink_RSL_MIN']
        # for i in itcol_name:
        #     na_bool_ind = np.isnan(entity.loc[:,i].values)
        #     nonzeros_ind = np.nonzero(entity.loc[:,i].values) ## include nans
        #     temp = np.zeros(len(entity.loc[:,i]))
        #     temp[nonzeros_ind] = 1
        #     temp = temp - na_bool_ind - na_bool_ind
        #     entity['%s_nan'%(i)] = temp
            
        ## set nans to 0
        entity = entity.fillna(0)
        
        ## make 3-d input
        temp_input.append(entity.drop(columns=['KEY','Time','Date']))
    
    temp_input = array(temp_input).reshape(-1,21)
    # ## min-max normalization
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = scaler.fit(temp_input)
    # temp_input = scaler.transform(temp_input)
    # temp_input = temp_input.reshape(-1,96,29)
    temp_input = temp_input.reshape(-1,3,32,21)
    return temp_input

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

reducelr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.001, cooldown=0, min_lr=0)
checkpoint = ModelCheckpoint(filepath='model/best_weights.h5',monitor='loss',mode='auto' ,save_best_only='True')
callback_lists = [reducelr,checkpoint]
# the dataset knows the number of features, 29
n_features = 29
# set time steps
n_steps = 96
# define model
model = Sequential()
model.add(LSTM(100, activation='sigmoid', return_sequences=True, input_shape=(n_steps, n_features), kernel_initializer='he_normal'))
model.add(LSTM(100, activation='sigmoid'))
#model.add(Dropout(0.2))
model.add(Dense(n_features))
model.add(Dense(2,activation='softmax'))
rmsprop = RMSprop(learning_rate=0.01)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


for date in all_date:
    label = out_label[out_label['Time']==date]
    x_temp = reconstruct_lstm_input(label)
    exec('x_{} = x_temp'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
    
    x_temp = pd.DataFrame(x_temp)
    exec("x_temp.to_csv('../all_data/existedge_data_2d_96_29/x_{}_96_29.csv')".format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))


for date in train_date:

    # convert into input/output
    train_out_label = out_label[out_label['Time']==date]
    exec('X = x_{}'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
    # X = reconstruct_lstm_input(train_out_label)
    y = to_categorical(train_out_label['label'].values)
    # fit model
    print("Training Modle. Date: ","{:s}.".format(date))
    if date == '2017/9/27':
        history = model.fit(X, y, epochs=100, verbose=2, callbacks=[checkpoint])
    else:
        model = load_model('model/best_weights.h5')
        history = model.fit(X, y, epochs=100, verbose=2, callbacks=[checkpoint])
    #exec('x_{} = X'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
    
    with open('trainwCallbackHistoryDict.txt', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

print('Training Completed!')

# demonstrate prediction
for date in test_date:
    test_out_label = out_label[out_label['Time']==date]
    #x_input = reconstruct_lstm_input(test_out_label)
    exec('x_input = x_{}'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
    y_out = test_out_label['label'].values
    x_input = x_input.reshape((-1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=1)
    print("Predictions for date: ", "{:s}.".format(date),
          "Model trained with data from", "{:s}".format(train_date[0]),
          "to", "{:s}.".format(train_date[len(train_date)-1]))
    roc_score, ap_score = evaluate_prediction(yhat, y_out)
    # exec('x_{} = x_input'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))


out_all = []
hat_all = []
for date in test_date:
    test_out_label = out_label[out_label['Time']==date]
    y_out = test_out_label['label'].values
    exec('x_input=x_{}'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
    yhat = model.predict(x_input, verbose=1)
    out_all.append(y_out)
    for i in range(len(yhat)):
        hat_all.append(yhat[i,:])
out_all = array(out_all)
out_all = array([i for j in out_all for i in j])
hat_all = array(hat_all)
evaluate_prediction(hat_all, out_all)
    
x_train = []
y_train = []
for date in train_date:
    exec('x_input=x_{}'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
    train_out_label = out_label[out_label['Time']==date]
    y = to_categorical(train_out_label['label'].values)
    for i in range(len(x_input)):
        x_train.append(x_input[i])
        y_train.append(y[i])
x_train = array(x_train)
y_train = array(y_train)

x_test = []
for date in test_date:
    exec('x_input=x_{}'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
    for i in range(len(x_input)):
        x_test.append(x_input[i])
x_test = array(x_test)

alltrain_reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
alltrain_checkpoint = ModelCheckpoint(filepath='model/alltrain_best_weights.h5',monitor='val_loss',mode='auto' ,save_best_only='True')
alltrain_callback_lists = [alltrain_reducelr,alltrain_checkpoint]


# define alldata_model
alldata_model = Sequential()
alldata_model.add(LSTM(100, activation='sigmoid', return_sequences=True, input_shape=(n_steps, n_features), kernel_initializer='he_normal'))
alldata_model.add(LSTM(100, activation='sigmoid'))
#model.add(Dropout(0.2))
alldata_model.add(Dense(n_features))
alldata_model.add(Dense(2,activation='softmax'))
rmsprop = RMSprop(learning_rate=0.01)
alldata_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
alldata_model.fit(x_train, y_train, epochs=100, verbose=2, 
                  validation_split=0.1, callbacks = alltrain_callback_lists)

alldata_model = load_model('model/alltrain_best_weights.h5')

yhat_all = alldata_model.predict(x_test, verbose=1)
evaluate_prediction(yhat_all, out_all)

for date in all_date:
    exec('x=x_{}'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
    x.tofile("../all_data/x_{}_3d_norm.bin".format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))

## recover from disk
x = np.fromfile("../all_data/x_0927_3d_norm.bin")
x = x.reshape(-1,96,29)

y = np.fromfile("../all_data/staticedge_date_3d_norm/x_0927_3d_norm_ae.bin")
y = y.reshape(-1,96,29)