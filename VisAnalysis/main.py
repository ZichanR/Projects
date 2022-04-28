#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 11:01:13 2021

@author: zichan
"""
import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import datetime as dt
import random
import multiprocessing as mp

from sklearn.model_selection import train_test_split
from integration import VisAnaintegration
from correction import VisAnacorrection
from transformation import VisAnatransformation#, VisAnaTrainTesttransformation
from visualization import VisAnavisualization
# from sklearn.svm import LinearSVC,SVC
# from sklearn.metrics import *
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

parser = argparse.ArgumentParser(description='PyTorch VisAnalysis')
parser.add_argument('--root_data', type=str, default='../all_data')
parser.add_argument('--root_image', type=str, default='./visualization')
parser.add_argument('--root_processed_data', type=str, default='./processedData')
parser.add_argument('--analyze', action='store_true',
                    help='process raw data')
parser.add_argument('--transform', action='store_true',
                    help='transform disregard train test')
parser.add_argument('--visualize', dest='visualize', action='store_true',
                    help='visualization mode')
parser.add_argument('--target_visualize_data', type=str, default='corrected', choices=['corrected','transformed','raw','pmads', 'pamds_embed', 'visAna_embed'])
parser.add_argument('--target_visualize_link', type=str, default='AAAI-aa-1---AATF-aa-1')
parser.add_argument('--target_visualize_date', type=str, default='2017/10/8')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate data in machin learning algorithms')
parser.add_argument('--evaluate_ml',type=str, default='RF', choices=['RF','SVM','DT','LR','DNN'],
                    help='evaluation methods')
parser.add_argument('--evaluate_data',type=str, default='visAna', choices=['visAna','pmads'],
                    help='evaluateion data')
parser.add_argument('--print_freq', type=int, default=100)

def main():
    args = parser.parse_args()
    if args.analyze:
        print("Start VisAnalysis")
        all_data = pd.read_csv('{}/all_data.csv'.format(args.root_data))
        ## get all labels
        pmads = pd.read_csv('{}/pmads_data.csv'.format(args.root_data))
        print("Raw data loaded")
        labelinfo = pmads.describe()
        out_label = pmads[['Time','KEY','label']]
        integrated_data, integrated_label = VisAnaintegration(args, all_data=all_data, label_data=out_label, save_intergrated_data=True, root_processed_data=args.root_processed_data)
        corrected_data = VisAnacorrection(integrated_data, save_corrected_data=True, root_processed_data=args.root_processed_data)
    
    if args.transform:
        if not 'corrected_data' in locals():
            filename = '%s/correctedlist.json' % (args.root_processed_data)
            corrected_data = []
            with open(filename, "r") as f:
                load_data = json.loads(f.read())
            for json_key_date_data in load_data:
                corrected_data.append(pd.DataFrame.from_dict(json_key_date_data))
            print("Corrected data loaded")
        transformed_data = VisAnatransformation(args, corrected_data, save_transformed_data = True, root_processed_data=args.root_processed_data, save_filename='transformedlist.json')
    
    # if args.trainTestTransform:
    #     root_processed_data = args.load_from_saved_trf_folder
    #     ## train/test traditional 8/2 split keep anomaly ratio
        
    #     if not 'corrected_data' in locals():
    #         filename = '%s/correctedlist.json' % (args.root_processed_data)
    #         corrected_data = []
    #         with open(filename, "r") as f:
    #             load_data = json.loads(f.read())
    #         for json_key_date_data in load_data:
    #             corrected_data.append(pd.DataFrame.from_dict(json_key_date_data))
    #         print("Corrected data loaded")
                
    #     if not 'integrated_label' in locals():
    #         label_filename = '%s/listlabel.json' % (args.root_processed_data)
    #         with open(label_filename, 'r') as f:
    #             label = [int(line.rstrip('\n')) for line in f]
    #         print("Labels loaded")
                
    #     train_d3_data, train_label, test_d3_data, test_label = VisAnaTrainTesttransformation(args, corrected_data, label,
    #                           save_transformed_data = True, root_processed_data=root_processed_data, 
    #                           save_train_filename='train_data.json', save_test_filename='test_data.json')
        
    if args.evaluate:
        if args.evaluate_data == 'visAna':
            # transform with train/test split
            if args.load_from_saved_trf_folder:
                traindata_filename = '%s/train_data.json' % (args.load_from_saved_trf)
                trainlabel_filename = '%s/train_label.json' % (args.load_from_saved_trf)
                testdata_filename = '%s/test_data.json' % (args.load_from_saved_trf)
                testlabel_filename = '%s/test_label.json' % (args.load_from_saved_trf)
                
                train_data = []
                with open(traindata_filename, "r") as f:
                    load_data = json.loads(f.read())
                for json_key_date_data in load_data:
                    train_data.append(pd.DataFrame.from_dict(json_key_date_data))
                with open(trainlabel_filename, 'r') as f:
                    train_label = [int(line.rstrip('\n')) for line in f]
                    
                test_data = []
                with open(testdata_filename, "r") as f:
                    load_data = json.loads(f.read())
                for json_key_date_data in load_data:
                    test_data.append(pd.DataFrame.from_dict(json_key_date_data))
                with open(testlabel_filename, 'r') as f:
                    test_label = [int(line.rstrip('\n')) for line in f]
                
        if args.evaluate_data == 'pmads':
            if not 'pmads' in locals():
                data = pd.read_csv('{}/pmads_data.csv'.format(args.root_data)).drop(['KEY','Time'],axis=1)
            else:
                data = pmads.drop(['KEY','Time'],axis=1)
            
    if args.visualize:
        print("Start visualization")
        if not 'integrated_label' in locals():
            label_filename = '%s/listlabel.json' % (args.root_processed_data)
            with open(label_filename, 'r') as f:
                label = [int(line.rstrip('\n')) for line in f]
            print("     labels loaded")
        else:
            label = integrated_label
            
        if args.target_visualize_data == 'corrected':
            if not 'corrected_data' in locals():
                filename = '%s/correctedlist.json' % (args.root_processed_data)
                visualize_data = []
                with open(filename, "r") as f:
                    load_data = json.loads(f.read())
                for json_key_date_data in load_data:
                    visualize_data.append(pd.DataFrame.from_dict(json_key_date_data))
                print("     corrected data loaded")
            else:
                visualize_data = corrected_data
            VisAnavisualization(args=args, visualize_data=visualize_data, out_label=label, root_image=args.root_image, visualize = 'preliminary')
            VisAnavisualization(args=args, visualize_data=visualize_data, out_label=label, root_image=args.root_image, visualize = 'feature_distribution')
            
        if args.target_visualize_data == 'transformed':
            if not 'transformed_data' in locals():
                filename = '%s/transformedlist.json' % (args.root_processed_data)
                visualize_data = []
                with open(filename, "r") as f:
                    load_data = json.loads(f.read())
                for json_key_date_data in load_data:
                    visualize_data.append(pd.DataFrame.from_dict(json_key_date_data))
                print("     transformed data loaded")
            else:
                visualize_data = transformed_data
            VisAnavisualization(args=args, visualize_data=visualize_data, out_label=label, root_image=args.root_image, visualize = 'preliminary')
            VisAnavisualization(args=args, visualize_data=visualize_data, out_label=label, root_image=args.root_image, visualize = 'feature_distribution')
            
        if args.target_visualize_data == 'pmads':
            if not 'pmads' in locals():
                visualize_data = pd.read_csv('{}/pmads_data.csv'.format(args.root_data))
            else:
                visualize_data = pmads
            print("     PMADS data loaded")
            VisAnavisualization(args=args, visualize_data=visualize_data, out_label=label, root_image=args.root_image, visualize = 'feature_distribution')
            VisAnavisualization(args=args, visualize_data=visualize_data, out_label=label, root_image=args.root_image, visualize = 'tsne')

        if args.target_visualize_data == 'raw':
            if not 'integrated_data' in locals():
                filename = '%s/intergratedlist.json' % (args.root_processed_data)
                visualize_data = []
                with open(filename, "r") as f:
                    load_data = json.loads(f.read())
                for json_key_date_data in load_data:
                    visualize_data.append(pd.DataFrame.from_dict(json_key_date_data))
                print("     raw_integrated data loaded")
            else:
                visualize_data = integrated_data
            VisAnavisualization(args=args, visualize_data=visualize_data, out_label=label, root_image=args.root_image, visualize = 'preliminary')
        
        if args.target_visualize_data == 'pamds_embed':
            # get inner layer data from dnn
            VisAnavisualization(args=args, visualize_data=visualize_data, out_label=label, root_image=args.root_image, visualize = 'preliminary')
        
        if args.target_visualize_data == 'visAna_embed':
            # get inner layer data from dnn
            VisAnavisualization(args=args, visualize_data=visualize_data, out_label=label, root_image=args.root_image, visualize = 'preliminary')

if __name__ == '__main__':
    main()

# ######## 128+128+stat
# trainNE = trainPMADS.loc[:,['Source_Node','Sink_Node']]
# trainentryNEn2v = []
# trainexitNEn2v = []
# for i in range(len(trainPMADS)):
#     entryNEn2v = np.delete(fn2vatt[np.where(fn2vatt[:,0] == trainNE.iloc[i,0])[0]],0,1).astype(float)
#     exitNEn2v = np.delete(fn2vatt[np.where(fn2vatt[:,0] == trainNE.iloc[i,1])[0]],0,1).astype(float)
#     trainentryNEn2v.append(entryNEn2v)
#     trainexitNEn2v.append(exitNEn2v)
# trainentryNEn2v = np.array(trainentryNEn2v)
# trainentryNEn2v.resize(len(trainPMADS),128)
# trainexitNEn2v = np.array(trainexitNEn2v)
# trainexitNEn2v.resize(len(trainPMADS),128)

# testNE = testPMADS.loc[:,['Source_Node','Sink_Node']]
# testentryNEn2v = []
# testexitNEn2v = []
# for i in range(len(testPMADS)):
#     entryNEn2v = np.delete(fn2vatt[np.where(fn2vatt[:,0] == testNE.iloc[i,0])[0]],0,1).astype(float)
#     exitNEn2v = np.delete(fn2vatt[np.where(fn2vatt[:,0] == testNE.iloc[i,1])[0]],0,1).astype(float)
#     testentryNEn2v.append(entryNEn2v)
#     testexitNEn2v.append(exitNEn2v)
# testentryNEn2v = np.array(testentryNEn2v)
# testentryNEn2v.resize(len(testPMADS),128)
# testexitNEn2v = np.array(testexitNEn2v)
# testexitNEn2v.resize(len(testexitNEn2v),128)

# X_train4 = np.hstack((trainentryNEn2v,trainexitNEn2v,trainPMADS.values[:,3:28]))
# y_train4 = trainPMADS.values[:,29].astype(np.float32)

# X_test4 = np.hstack((testentryNEn2v,testexitNEn2v,testPMADS.values[:,3:28]))
# y_test4 = testPMADS.values[:,29].astype(np.float32)

# rdfrt = RandomForestClassifier()
# rdfrt.fit(X_train4, y_train4)

# y_pred4 = rdfrt.predict(X_test4)

# macro_f1 = f1_score(y_test4, y_pred4, average="macro")
# micro_f1 = f1_score(y_test4, y_pred4, average="micro")
# print(classification_report(y_test4, y_pred4))
# print(confusion_matrix(y_test4, y_pred4))


# ### W node2vec
# X_train5 = np.hstack((trainEn2v,trainPMADS.values[:,5:24]))
# y_train5 = trainPMADS.values[:,24].astype(np.int16)

# X_test5 = np.hstack((testEn2v,testPMADS.values[:,5:24]))
# y_test5 = testPMADS.values[:,24].astype(np.int16)

# ### w/o node2vec
# X_train5 = trainPMADS.values[:,5:24]
# y_train5 = trainPMADS.values[:,24].astype(np.int16)

# X_test5 = testPMADS.values[:,5:24]
# y_test5 = testPMADS.values[:,24].astype(np.int16)

# rdfrt = RandomForestClassifier()
# rdfrt.fit(X_train5, y_train5)
# y_pred_rdfrt = rdfrt.predict(X_test5)

# gbdc = GradientBoostingClassifier()
# gbdc.fit(X_train5, y_train5)
# y_pred_gbdc = gbdc.predict(X_test5)

# svm = SVC()
# svm.fit(X_train5, y_train5)
# y_pred_svm = svm.predict(X_test5)

# lr = LogisticRegression()
# lr.fit(X_train5, y_train5)
# y_pred_lr = lr.predict(X_test5)

# dt = DecisionTreeClassifier()
# dt.fit(X_train5, y_train5)
# y_pred_dt = dt.predict(X_test5)

# y_pred5 = y_pred_dt

# macro_f1 = f1_score(y_test5, y_pred5, average="macro")
# micro_f1 = f1_score(y_test5, y_pred5, average="micro")
# print(classification_report(y_test5, y_pred5))
# print(confusion_matrix(y_test5, y_pred5))