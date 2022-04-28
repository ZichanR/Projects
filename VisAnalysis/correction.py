#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 18:30:09 2021

@author: zichan
"""
import datetime as dt
import pandas as pd
import json
import os
import time
import multiprocessing as mp

def VisAnacorrection(args, integrated_data, save_corrected_data=True, root_processed_data='./processedData'):
    
    if not os.path.exists(root_processed_data):
        os.mkdir(root_processed_data)
    print("Start data correction")
    total = len(integrated_data)
    start_time = time.time()
    
    pool = mp.Pool(min(mp.cpu_count(),26))
    corrected_data = pool.starmap(thread_correction, [(count_entity, key_date_data, total, args) for count_entity, key_date_data in enumerate(integrated_data)])
    pool.close()

    print("End data correction")
    print("Total correction time {:.3f}s".format(time.time()-start_time))
    
    if save_corrected_data:
        filename = '%s/correctedlist.json' % (root_processed_data)
        saving_data = []
        for crtd in corrected_data:
            saving_data.append(json.loads(crtd.to_json()))
        with open(filename, "w") as fp:
            json.dump(saving_data, fp)
        # reload_data = []
        # with open("./processedData/correctedlist.json", "r") as f:
        #     load_data = json.loads(f.read())
        # for json_key_date_data in load_data:
        #     reload_data.append(pd.DataFrame.from_dict(json_key_date_data))
        print("Save data correction as {}".format(filename))
    return corrected_data

def thread_correction(count_entity, key_date_data, total, args):
    # corrected_data = []
    start_time = time.time()
    
    key_date_data = integrateDuplicatedDate(key_date_data)
    key_date_data = estimateMissingData(key_date_data)
    
    # corrected_data.append(key_date_data)
    k = key_date_data.loc[0,'KEY']
    d = dt.datetime.strptime(key_date_data.loc[0,'Time'], '%Y-%m-%d %H:%M:%S').date()
    if count_entity % args.print_freq == 0:
        print("    Pool Correction {}/{} | Ave. Time {:.3f}s |Entity {} {}".format(count_entity, total, (time.time()-start_time)/args.print_freq, k, d.strftime('%Y/%m/%d')))
    return key_date_data #corrected_data

def integrateDuplicatedDate(key_date_data):
    if len(key_date_data) > 96:
        datetimes = key_date_data['Time'].unique()
        datetimes = [dt.datetime.strptime(datetimes[i], '%Y-%m-%d %H:%M:%S') for i in range(len(datetimes))]
        d = datetimes[0].date()
        times = list(set([datetimes[i].time() for i in range(len(datetimes))]))
        times.sort()
        corrected_data = pd.DataFrame(columns=key_date_data.columns)
        for t in times:
            dts = dt.datetime.combine(d,t).strftime('%Y-%m-%d %H:%M:%S')
            key_date_time_data = key_date_data.loc[key_date_data['Time'] == dts]
            if len(key_date_time_data) > 1:
                corrected_time_data = pd.DataFrame(columns=key_date_data.columns)
                corrected_time_data.loc[0] = key_date_time_data.mean(axis=0, skipna=True, numeric_only=True)
                corrected_time_data.loc[0,'KEY'] = key_date_data.loc[0,'KEY']
                corrected_time_data.loc[0,'Time'] = dts
                key_date_time_data = corrected_time_data
            corrected_data = corrected_data.append(key_date_time_data, ignore_index=True)
    else:
        corrected_data = key_date_data
    return corrected_data

def estimateMissingData(key_date_data):
    min_SNR = 0
    min_RSL = -90
    min_RSL_thre = -9999
    replace_error_indicators = -1
    corrected_data = pd.DataFrame(columns=key_date_data.columns)
    aid_row_head = pd.DataFrame([key_date_data.iloc[1,:]], index=[-1])
    aid_row_tail = pd.DataFrame([key_date_data.iloc[94,:]], index=[96])
    key_date_data = key_date_data.append([aid_row_head,aid_row_tail]).sort_index()
    for f in range(key_date_data.shape[1]):
        feature_vector = key_date_data.iloc[:,f].to_frame()
        corrected_feature_vector = pd.DataFrame(columns=feature_vector.columns)
        replace_values = replace_error_indicators
        if "SNR"  in feature_vector.columns[0]:
            replace_values = min_SNR
        if "RSL" in feature_vector.columns[0]:
            replace_values = min_RSL
            if "thre" in feature_vector.columns[0]:
                replace_values = min_RSL_thre
        if "KEY" in feature_vector.columns[0] or "Time" in feature_vector.columns[0]:
            corrected_feature_vector = key_date_data.iloc[1:97,f].to_frame()
        else:
            for i in range(0,96):
                if pd.isna(feature_vector.loc[i-1:i+1]).values.all():
                    corrected_feature_vector.loc[i] = replace_values
                elif not pd.isna(feature_vector.loc[i]).values[0]:
                    corrected_feature_vector.loc[i] = feature_vector.loc[i]
                else:
                    corrected_feature_vector.loc[i] = feature_vector.loc[i-1:i+1].mean(axis=0, skipna=True, numeric_only=True)
                if pd.isna(key_date_data.iloc[i+1,2:23]).values.all() and (pd.isna(key_date_data.iloc[i-1:i+1,2:23]).values.all() or pd.isna(key_date_data.iloc[i+2:i+4,2:23]).values.all()):
                    corrected_feature_vector.loc[i] = replace_values
        corrected_data.iloc[:,f] = corrected_feature_vector.values.reshape(-1,)
    return corrected_data
    