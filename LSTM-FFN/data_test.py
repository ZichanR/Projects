#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:28:11 2020

@author: zichan
"""
import os
import numpy as np
import pandas as pd
import datetime
import random

from sklearn.model_selection import train_test_split
# from sklearn.svm import LinearSVC,SVC
# from sklearn.metrics import *
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


pcdPMADS = pd.read_csv('../data/pmads_data.csv')
out_label = pcdPMADS[['Time','KEY','label']]
all_edge = pcdPMADS['KEY'].values
#### delete repeated KEYs
all_edge = pd.DataFrame()
all_edge['KEY'] = np.unique(pcdPMADS['KEY'].values)
# add source_node and sink_node
all_edge['Source'] = np.array([i[0:4] for i in all_edge['KEY']])
all_edge['Sink'] = np.array([i[12:16] for i in all_edge['KEY']])
all_edge = all_edge.values
np.save("../all_data/static_all_edge.edge", all_edge)

# os.makedirs("../all_data/staticedge_date_3d_norm")

all_date = ['2017/9/27','2017/9/28','2017/9/29','2017/9/30','2017/10/1','2017/10/2','2017/10/3',
              '2017/10/4','2017/10/5','2017/10/6','2017/10/7','2017/10/8','2017/10/9','2017/10/10',
              '2017/10/11','2017/10/12','2017/10/13','2017/10/14','2017/10/15','2017/10/16',
              '2017/10/17','2017/10/18','2017/10/19','2017/10/20','2017/10/21','2017/10/22']

### create featrue matrixes for edges does not exist in current date
for date in all_date:
    x_temp = []
    exec('x_input=np.fromfile("../all_data/x_{}_3d_norm.bin").reshape(-1,96,29)'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
    x_input_keys = out_label[out_label['Time']==date]['KEY'].values
    for key_ind in range(len(all_edge)):
        if all_edge[key_ind,0] in x_input_keys:
            x_temp.append(x_input[np.where(x_input_keys==all_edge[key_ind,0]),:,:].reshape(96,29))
        else:
            x_temp.append(np.zeros([96,29]))
    x_temp = np.array(x_temp)
    exec('x_temp.tofile("../all_data/staticedge_date_3d_norm/x_{}_3d_norm_ae.bin")'.format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))
    # x.tofile("../all_data/x_{}_3d_norm.bin".format(datetime.datetime.strptime(date,'%Y/%m/%d').date().strftime('%m%d')))

    
def MinMaxNormalization(x,Min,Max):
    x = (x - Min) / (Max - Min)
    return x

tPMADS = pcdPMADS.T.values

nmlPMADS = []
ridx = 0
for row in tPMADS:
    if ridx > 1:
        nrow = []
        for item in row:
            n = MinMaxNormalization(item, np.min(row), np.max(row))
            nrow.append(n)
        nmlPMADS.append(nrow)
    ridx += 1

nmlPMADS = np.array(nmlPMADS)
nmlPMADS = np.row_stack((tPMADS[0:2,], nmlPMADS)).T

X = nmlPMADS[:,2:21]
y = nmlPMADS[:,21].astype(np.int16)

Xorg = pcdPMADS.values[:,2:21]
yorg = pcdPMADS.values[:,21].astype(np.int16)

X_train, X_test, y_train, y_test = train_test_split(Xorg, yorg, test_size=(1-0.056), random_state=2018)

rdfrt = RandomForestClassifier()
rdfrt.fit(X_train, y_train)

y_pred = rdfrt.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average="macro")
micro_f1 = f1_score(y_test, y_pred, average="micro")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


## seprate dataset by date into train and test

trainPMADS = pd.read_csv('../data/trainSepPMADS.csv')
testPMADS = pd.read_csv('../data/testSepPMADS.csv')

### active learning training set
trainPMADS_p = trainPMADS[trainPMADS['label'] == 1].values
trainPMADS_n = trainPMADS[trainPMADS['label'] == 0].values
num_train_n = x = [[i] for i in range(len(trainPMADS_n))]
random.shuffle(num_train_n)
trainPMADS_n = trainPMADS_n[num_train_n[0:len(trainPMADS_p)]]
trainPMADS_n.resize(len(trainPMADS_p),25)
trainPMADS_act = np.vstack((trainPMADS_p,trainPMADS_n))
#### trainPMADS replace trainPMADS
trainPMADS = pd.DataFrame(trainPMADS_act)
trainPMADS.columns = testPMADS.columns.tolist()

#### hash + hash + stat
trainNE = trainPMADS.loc[:,['EntryNE','ExitNE']]

for i in range(len(trainPMADS)):
    trainNE.loc[i, 'EntryNE'] = hash(trainNE.loc[i,'EntryNE'])
    trainNE.loc[i, 'ExitNE'] = hash(trainNE.loc[i,'ExitNE'])

testNE = testPMADS.loc[:,['EntryNE','ExitNE']]

for i in range(len(testPMADS)):
    testNE.loc[i, 'EntryNE'] = hash(testNE.loc[i,'EntryNE'])
    testNE.loc[i, 'ExitNE'] = hash(testNE.loc[i,'ExitNE'])

X_train2 = np.hstack((trainNE,trainPMADS.values[:,5:24]))
y_train2 = trainPMADS.values[:,24].astype(np.int16)

X_test2 = np.hstack((testNE,testPMADS.values[:,5:24]))
y_test2 = testPMADS.values[:,24].astype(np.int16)

rdfrt = RandomForestClassifier()
rdfrt.fit(X_train2, y_train2)

y_pred2 = rdfrt.predict(X_test2)

macro_f1 = f1_score(y_test2, y_pred2, average="macro")
micro_f1 = f1_score(y_test2, y_pred2, average="micro")
print(classification_report(y_test2, y_pred2))
print(confusion_matrix(y_test2, y_pred2))

#### n2v result
nodeemb_file = open('n2vemb/allnode.nodeemb', 'r')
fn2v = nodeemb_file.readlines()
fn2v.pop(0)
node_num = 1854
attribute_number = 128
attributes = []
for line in fn2v:
    attribute = []
    node1 = str(line.split(' ')[0].strip())
    attribute.append(node1)
    for i in range(1,129):
        attribute1 = float(line.split(' ')[i].strip())
        attribute.append(attribute1)
    attributes.append(attribute)
fn2vatt = np.matrix(attributes)

###### 128 + stat
trainNE = trainPMADS.loc[:,['EntryNE','ExitNE']]
trainentryNEn2v = []
for i in range(len(trainPMADS)):
    entryNEn2v = np.delete(fn2vatt[np.where(fn2vatt[:,0] == trainNE.loc[i, 'EntryNE'])[0]],0,1).astype(float)
    exitNEn2v = np.delete(fn2vatt[np.where(fn2vatt[:,0] == trainNE.loc[i, 'ExitNE'])[0]],0,1).astype(float)
    trainentryNEn2v.append(entryNEn2v-exitNEn2v)
trainentryNEn2v = np.array(trainentryNEn2v)
trainentryNEn2v.resize(len(trainPMADS),128)


testNE = testPMADS.loc[:,['EntryNE','ExitNE']]
testentryNEn2v = []
for i in range(len(testPMADS)):
    entryNEn2v = np.delete(fn2vatt[np.where(fn2vatt[:,0] == trainNE.loc[i, 'EntryNE'])[0]],0,1).astype(float)
    exitNEn2v = np.delete(fn2vatt[np.where(fn2vatt[:,0] == trainNE.loc[i, 'ExitNE'])[0]],0,1).astype(float)
    testentryNEn2v.append(entryNEn2v-exitNEn2v)
testentryNEn2v = np.array(testentryNEn2v)
testentryNEn2v.resize(len(testPMADS),128)


X_train3 = np.hstack((trainentryNEn2v,trainPMADS.values[:,5:24]))
y_train3 = trainPMADS.values[:,24].astype(np.int16)

X_test3 = np.hstack((testentryNEn2v,testPMADS.values[:,5:24]))
y_test3 = testPMADS.values[:,24].astype(np.int16)

rdfrt = RandomForestClassifier()
rdfrt.fit(X_train3, y_train3)

y_pred3 = rdfrt.predict(X_test3)

macro_f1 = f1_score(y_test3, y_pred3, average="macro")
micro_f1 = f1_score(y_test3, y_pred3, average="micro")
print(classification_report(y_test3, y_pred3))
print(confusion_matrix(y_test3, y_pred3))

######## 128+128+stat
trainNE = trainPMADS.loc[:,['EntryNE','ExitNE']]
trainentryNEn2v = []
trainexitNEn2v = []
for i in range(len(trainPMADS)):
    entryNEn2v = np.delete(fn2vatt[np.where(fn2vatt[:,0] == trainNE.loc[i, 'EntryNE'])[0]],0,1).astype(float)
    exitNEn2v = np.delete(fn2vatt[np.where(fn2vatt[:,0] == trainNE.loc[i, 'ExitNE'])[0]],0,1).astype(float)
    trainentryNEn2v.append(entryNEn2v)
    trainexitNEn2v.append(exitNEn2v)
trainentryNEn2v = np.array(trainentryNEn2v)
trainentryNEn2v.resize(len(trainPMADS),128)
trainexitNEn2v = np.array(trainexitNEn2v)
trainexitNEn2v.resize(len(trainPMADS),128)

testNE = testPMADS.loc[:,['EntryNE','ExitNE']]
testentryNEn2v = []
testexitNEn2v = []
for i in range(len(testPMADS)):
    entryNEn2v = np.delete(fn2vatt[np.where(fn2vatt[:,0] == testNE.loc[i, 'EntryNE'])[0]],0,1).astype(float)
    exitNEn2v = np.delete(fn2vatt[np.where(fn2vatt[:,0] == testNE.loc[i, 'ExitNE'])[0]],0,1).astype(float)
    testentryNEn2v.append(entryNEn2v)
    testexitNEn2v.append(exitNEn2v)
testentryNEn2v = np.array(testentryNEn2v)
testentryNEn2v.resize(len(testPMADS),128)
testexitNEn2v = np.array(testexitNEn2v)
testexitNEn2v.resize(len(testexitNEn2v),128)

X_train4 = np.hstack((trainentryNEn2v,trainexitNEn2v,trainPMADS.values[:,5:24]))
y_train4 = trainPMADS.values[:,24].astype(np.int16)

X_test4 = np.hstack((testentryNEn2v,testexitNEn2v,testPMADS.values[:,5:24]))
y_test4 = testPMADS.values[:,24].astype(np.int16)

rdfrt = RandomForestClassifier()
rdfrt.fit(X_train4, y_train4)

y_pred4 = rdfrt.predict(X_test4)

macro_f1 = f1_score(y_test4, y_pred4, average="macro")
micro_f1 = f1_score(y_test4, y_pred4, average="micro")
print(classification_report(y_test4, y_pred4))
print(confusion_matrix(y_test4, y_pred4))

##### check the difference between pred and test
truePosid = np.array(np.where(y_test4 == 1))
truePosid.resize(truePosid.shape[1],)
trueNegid = np.array(np.where(y_test4 == 0))
trueNegid.resize(trueNegid.shape[1],)
predPosid = np.array(np.where(y_pred4 == 1))
predPosid.resize(predPosid.shape[1],)
predNegid = np.array(np.where(y_pred4 == 0))
predNegid.resize(predNegid.shape[1],)
fposid = list(set(predPosid).difference(set(truePosid)))
fnegid = list(set(predNegid).difference(set(trueNegid)))
fPos = np.hstack((np.array(fposid).reshape(len(fposid),1), testPMADS.values[fposid]))
fNeg = np.hstack((np.array(fnegid).reshape(len(fnegid),1), testPMADS.values[fnegid]))
names = testPMADS.columns.values.tolist()
names.insert(0,'orig_index')
fPos = pd.DataFrame(columns=names, data=fPos)
fNeg = pd.DataFrame(columns=names, data=fNeg)
fPos.to_csv('detailed_info/falsePos3.csv')
fNeg.to_csv('detailed_info/falseNeg3.csv')

### l1 + stat
trainNE = trainPMADS.loc[:,['EntryNE','ExitNE']]
trainentryNEn2v = []
for i in range(len(trainPMADS)):
    entryNEn2v = np.delete(fn2vatt[np.where(fn2vatt[:,0] == trainNE.loc[i, 'EntryNE'])[0]],0,1).astype(float)
    exitNEn2v = np.delete(fn2vatt[np.where(fn2vatt[:,0] == trainNE.loc[i, 'ExitNE'])[0]],0,1).astype(float)
    trainentryNEn2v.append(entryNEn2v-exitNEn2v)
trainentryNEn2v = np.array(trainentryNEn2v)
trainentryNEn2v.resize(len(trainPMADS),128)
trainEn2v = []
for j in range(len(trainPMADS)):
        trainEn2v.append(np.dot(np.abs(trainentryNEn2v[j]), np.ones(trainentryNEn2v[j].size)))
trainEn2v = np.array(trainEn2v)
trainEn2v.resize(len(trainEn2v),1)

testNE = testPMADS.loc[:,['EntryNE','ExitNE']]
testentryNEn2v = []
for i in range(len(testPMADS)):
    entryNEn2v = np.delete(fn2vatt[np.where(fn2vatt[:,0] == testNE.loc[i, 'EntryNE'])[0]],0,1).astype(float)
    exitNEn2v = np.delete(fn2vatt[np.where(fn2vatt[:,0] == testNE.loc[i, 'ExitNE'])[0]],0,1).astype(float)
    testentryNEn2v.append(entryNEn2v-exitNEn2v)
testentryNEn2v = np.array(testentryNEn2v)
testentryNEn2v.resize(len(testPMADS),128)
testEn2v = []
for j in range(len(testPMADS)):
        testEn2v.append(np.dot(np.abs(testentryNEn2v[j]), np.ones(testentryNEn2v[j].size)))
testEn2v = np.array(testEn2v)
testEn2v.resize(len(testEn2v),1)

X_train5 = np.hstack((trainEn2v,trainPMADS.values[:,5:24]))
y_train5 = trainPMADS.values[:,24].astype(np.int16)

X_test5 = np.hstack((testEn2v,testPMADS.values[:,5:24]))
y_test5 = testPMADS.values[:,24].astype(np.int16)

rdfrt = RandomForestClassifier()
rdfrt.fit(X_train5, y_train5)

y_pred5 = rdfrt.predict(X_test5)

macro_f1 = f1_score(y_test5, y_pred5, average="macro")
micro_f1 = f1_score(y_test5, y_pred5, average="micro")
print(classification_report(y_test5, y_pred5))
print(confusion_matrix(y_test5, y_pred5))

##### HadamardEmbedder n2v + stat

label_file = open("PMADS_dis_sta/groundtruth/label0928.label", 'r')
labels = label_file.readlines()
label_line = []
for line in labels:
    node1 = int(line.split('\t')[0].strip())
    node2 = int(line.split('\t')[1].strip())
    line = line.split('\t')
    line[len(line)-1] = line[len(line)-1].strip()
    label_line.append(line)
label = np.matrix(label_line).astype(int)

edge_attrs = []
for edge in label:
    edge_att = []
    node1_att = []
    node2_att = []
    for node in fn2vatt:
        if int(node[0,0]) == edge[0,0]:
            node1_att = node[:,1:129]
        if int(node[0,0]) == edge[0
            node2_att = node[:,1:129]
    if type(node1_att)
    edge_att = node1_att - node2_att
    edge_attrs.append(edge_att)
            









