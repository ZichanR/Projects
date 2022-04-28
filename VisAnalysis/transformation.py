#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 17:00:50 2021

@author: zichan
"""
import os
import datetime as dt
import pandas as pd
import json
import time
import random
import multiprocessing as mp

ml_mean = ['KEY','Time', 41.71777009046142, 41.84810567479463, 40.80764011399103, 40.95057788863914, 
           137.64992382830002, 121.40769344919845, -35.92477774992904, -36.04408215119809, 
           -37.06761100084376, -37.22448057850674,  -34.6249868768797, 822.5628805788381, 
           823.5016095534787, 11.616116809906663, 12.82038210000515, 37.748383285660985, 
           35.53220000604979, 41.72621119600941, 41.86209927791734, -35.94916849781282, 
           -36.058045948784084]
ml_std = ['KEY','Time', 5.395874254337868, 4.774100968857864, 6.188324676140016, 5.612063943339526, 
          146.4195626021331, 144.37401170201002, 6.1992456205726025, 6.0237847287564925, 
          7.005730960256092, 6.9411819904968866, 10.62427316491025, 237.14670272690603, 
          235.22267292260588, 20.134514489857516, 31.371472730910366, 107.88662946017642, 
          94.69800439083676, 5.356243327518758, 4.741434695314131, 6.110152724736332, 
          5.937808142858194]
ml_min = ['KEY','Time', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -90.0, -90.0, -90.0, -90.0, -90.0, 
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -90.0, -90.0]
ml_max = ['KEY','Time', 48.3, 48.3, 47.5, 47.4, 315.0, 315.0, -18.9, -18.3, -20.0, -20.0, -20.0, 
          900.0, 900.0, 264.0, 353.0, 900.0, 900.0, 48.2, 48.2, -21.5, -20.0]

def VisAnatransformation(args, corrected_data, save_transformed_data = True, 
                         root_processed_data='./processedData', save_filename='transformedlist.json'):
    
    print("Start data transformation")
    if not os.path.exists(root_processed_data):
        os.mkdir(root_processed_data)
        print("Root folder {} created.".format(root_processed_data))
    start_time = time.time()
    entity_time = start_time
    
    total_entity = len(corrected_data)

    feature_names = pd.DataFrame(corrected_data[0].columns)
    feature_names = feature_names.drop([0,1])
    SNR_index_list = []
    RSL_index_list = []
    EI_index_list = []
    for index in feature_names.index:
        if 'SNR' in feature_names.loc[index,0]:
            SNR_index_list.append(index)
        elif 'RSL' in feature_names.loc[index,0]:
            RSL_index_list.append(index)
        else:
            EI_index_list.append(index)
    
    pool = mp.Pool(min(mp.cpu_count(),26))
    transformed_data = pool.starmap(thread_transformation, 
                            [(args, eleid, tdfe, total_entity, SNR_index_list, RSL_index_list, EI_index_list, feature_names) 
                             for eleid, tdfe in enumerate(corrected_data)])
    pool.close()
    
    
    print("End data transformation")
    print("Total transformation time {:.3f}s".format(time.time()-start_time))
    # save transformed data
    if save_transformed_data:
        filename = '%s/%s' % (root_processed_data, save_filename)
        saving_data = []
        for tsfd in transformed_data:
            saving_data.append(json.loads(tsfd.to_json()))
        with open(filename, "w") as fp:
            json.dump(saving_data, fp)
        print("Save transformation data as {}".format(filename))
    return transformed_data

def thread_transformation(args, eleid, tdfe, total_entity, SNR_index_list, RSL_index_list, EI_index_list, feature_names):
    entity_time=time.time()
    tdfe.index = pd.RangeIndex(len(tdfe.index))
    transformed_data_ele = pd.DataFrame(columns=tdfe.columns)
    transformed_data_ele['KEY'] = tdfe['KEY']
    transformed_data_ele['Time'] = tdfe['Time']
    k = transformed_data_ele['KEY'][0]
    d = transformed_data_ele['Time'][0].split(" ")[0]
    for i,f in enumerate(SNR_index_list+RSL_index_list+EI_index_list):
        feature_vector = tdfe.iloc[:,f]
        if f in (SNR_index_list+RSL_index_list):
            # feature_vector = d2_data.iloc[:,f]
            if feature_names.loc[f,0] == 'RSL_thre':
                feature_vector.transform(lambda n: max(-90, min(n, -20)))
            transformed_data_ele.iloc[:,f] = feature_vector.transform(lambda n: (n - ml_mean[f]) / ml_std[f])
        if f in EI_index_list:
            transformed_data_ele.iloc[:,f] = feature_vector.transform(lambda n: (n - ml_min[f]) / (ml_max[f] - ml_min[f]) * (1 - (-1)) + (-1))
    if eleid % args.print_freq == 0:
        print("    Pool Transformation {}/{}| Ave. Time {:.3f}s |Entity {} {}".format(eleid, total_entity, (time.time()-entity_time)/args.print_freq, k, d))    
    return transformed_data_ele

# def VisAnaTrainTesttransformation(args, corrected_data, label,
#                                   save_transformed_data = True, root_processed_data='./processedData', 
#                                   save_train_filename='train_data.json', save_test_filename='test_data.json'):
#     print("Start train test transformation")
#     if not os.path.exists(root_processed_data):
#         os.mkdir(root_processed_data)
#         print("Root folder {} created.".format(root_processed_data))
#     start_time = time.time()
    
#     # step1 merge 3-d to 2-d for a global view of each feature
#     print("Start train test splitting")
#     label = pd.Series(label, name = 'label')
#     index_0 = label.index[label == 0].tolist()
#     index_1 = label.index[label == 1].tolist()
#     random.shuffle(index_0)
#     random.shuffle(index_1)
#     train_idx = index_0[:int(len(index_0)*0.8)] + index_1[:int(len(index_1)*0.8)]
#     test_idx = index_0[int(len(index_0)*0.8):] + index_1[int(len(index_1)*0.8):]
    
#     train=[corrected_data[idx] for idx in train_idx]
#     train_label = [corrected_data[idx] for idx in train_idx]
#     test=[corrected_data[idx] for idx in test_idx]
#     test_label = [corrected_data[idx] for idx in test_idx]
    
#     print("Start load data as a 2d matrix")
#     train_d2_data = pd.DataFrame(columns=train[0].columns)
#     num_train_entity = len(train)
#     for df in train:
#         df.index = pd.RangeIndex(len(df.index))
#         train_d2_data = train_d2_data.append(df,ignore_index=True)
#     test_d2_data = pd.DataFrame(columns=test[0].columns)
#     num_test_entity = len(test)
#     for df in test:
#         df.index = pd.RangeIndex(len(df.index))
#         test_d2_data = test_d2_data.append(df,ignore_index=True)

#     # step2 decide tranformation for each feature
#     # fit scaler on training data
#     feature_names = pd.DataFrame(corrected_data[0].columns)
#     feature_names = feature_names.drop([0,1])
#     SNR_index_list = []
#     RSL_index_list = []
#     EI_index_list = []
#     for index in feature_names.index:
#         if 'SNR' in feature_names.loc[index,0]:
#             SNR_index_list.append(index)
#         elif 'RSL' in feature_names.loc[index,0]:
#             RSL_index_list.append(index)
#         else:
#             EI_index_list.append(index)
            
#     transformed_train_data = pd.DataFrame(columns=train_d2_data.columns)
#     transformed_train_data['KEY'] = train_d2_data['KEY']
#     transformed_train_data['Time'] = train_d2_data['Time']
#     transformed_test_data = pd.DataFrame(columns=test_d2_data.columns)
#     transformed_test_data['KEY'] = test_d2_data['KEY']
#     transformed_test_data['Time'] = test_d2_data['Time']
    
#     pool = mp.Pool(min(mp.cpu_count(),21))
#     i, transformed_train_features, transformed_test_features = zip(*pool.starmap(thread_traintest_transformation, 
#                                                             [(i, f, train_d2_data.iloc[:,f], test_d2_data.iloc[:,f], SNR_index_list, RSL_index_list, EI_index_list, feature_names) 
#                                                              for i,f in enumerate(SNR_index_list+RSL_index_list+EI_index_list)]))
#     pool.close()
    
#     for fi, fvtrain, fvtest in zip(i, transformed_train_features, transformed_test_features):
#         transformed_train_data[feature_names.loc[fi,0]] = fvtrain
#         transformed_test_data[feature_names.loc[fi,0]] = fvtest
        
#     # step3 split 2-d to 3-d
#     print("Start split data as a list of 2d matrixes")
#     train_d3_data = []
#     for idx in range(num_train_entity):
#         train_d3_data.append(transformed_train_data.iloc[idx*96:96*(idx+1),:].reset_index(drop=True))
#     test_d3_data = []
#     for idx in range(num_test_entity):
#         test_d3_data.append(transformed_test_data.iloc[idx*96:96*(idx+1),:].reset_index(drop=True))
#     print("End data transformation")
#     print("Total transformation time {:.3f}s".format(time.time()-start_time))
#     # save transformed data
#     if save_transformed_data:
#         train_data_filename = '%s/%s' % (root_processed_data, save_train_filename)
#         saving_train_data = []
#         for tsfd in enumerate(train_d3_data):
#             saving_train_data.append(json.loads(tsfd.to_json()))
#         with open(train_data_filename, "w") as fp:
#             json.dump(saving_train_data, fp)
#         test_data_filename = '%s/%s' % (root_processed_data, save_test_filename)
#         saving_test_data = []
#         for tsfd in (test_d3_data):
#             saving_test_data.append(json.loads(tsfd.to_json()))
#         with open(test_data_filename, "w") as fp:
#             json.dump(saving_test_data, fp)
#         train_label_filename = '%s/train_label.json' % (root_processed_data)
#         with open(train_label_filename, 'w') as f:
#             for l in train_label:
#                 f.write(str(l) + '\n')
                
#         test_label_filename = '%s/test_label.json' % (root_processed_data)
#         with open(test_label_filename, 'w') as f:
#             for l in test_label:
#                 f.write(str(l) + '\n')
#         print("Save train test data transformation at folder {}".format(root_processed_data))
#     return train_d3_data, train_label, test_d3_data, test_label

# def thread_traintest_transformation(i, f, train_feature_vector, test_feature_vector, SNR_index_list, RSL_index_list, EI_index_list, feature_names):
#     entity_time = time.time()
#     if f in (SNR_index_list+RSL_index_list):
#         # feature_vector = d2_data.iloc[:,f]
#         if feature_names.loc[f,0] == 'RSL_thre':
#             train_feature_vector.transform(lambda n: max(-90, min(n, -20)))
#             test_feature_vector.transform(lambda n: max(-90, min(n, -20)))
#         norm = StandardScaler().fit(train_feature_vector.values.reshape(-1,1))
#         train_feature_vector = norm.transform(train_feature_vector.values.reshape(-1,1))
#         test_feature_vector = norm.transform(test_feature_vector.values.reshape(-1,1))
#     if f in EI_index_list:
#         norm = MinMaxScaler(feature_range=(-1, 1)).fit(train_feature_vector.values.reshape(-1,1))
#         train_feature_vector = norm.transform(train_feature_vector.values.reshape(-1,1))
#         test_feature_vector = norm.transform(test_feature_vector.values.reshape(-1,1))
#     print("    Transformation {}/21| {} |Time {:.3f}s".format(i+1, feature_names.loc[f,0], time.time()-entity_time))
#     return f, train_feature_vector, test_feature_vector

# lambda n: (n - u=mean) / s=std

# lambda n: (n - min=min) / (max=max - min=min) * (rang_max=1 - (rang_min=-1)) + (rang_min=-1)