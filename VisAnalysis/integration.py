#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 17:40:53 2021

@author: zichan
"""
import os
import datetime as dt
import pandas as pd
import json
import time
import multiprocessing as mp

def VisAnaintegration(args, all_data, label_data, save_intergrated_data=True, root_processed_data='./processedData'):
    
    if not os.path.exists(root_processed_data):
        os.mkdir(root_processed_data)
    print("Start data integration")
    keys = list(all_data['KEY'].unique())
    datetimes = all_data['Time'].unique()
    datetimes = [dt.datetime.strptime(datetimes[i], '%Y-%m-%d %H:%M:%S') for i in range(len(datetimes))]
    dates = list(set([datetimes[i].date() for i in range(len(datetimes))]))
    times = list(set([datetimes[i].time() for i in range(len(datetimes))]))
    dates.sort()
    times.sort()
    pool_total = len(keys)
    start_time = time.time()
    ## parallel processing
    pool = mp.Pool(min(mp.cpu_count(),26))
    integrated_data_lists, integrated_label_lists = zip(*pool.starmap(thread_integration, [(kid, dates, k, times, pool_total, all_data.loc[all_data['KEY'] == k], label_data) for kid, k in enumerate(keys)]))    
    pool.close()
    ## formating integrated_data integrated_label
    formate_time = time.time()
    integrated_data = []
    for dl in integrated_data_lists:
        for kd in dl:
            integrated_data.append(kd)
    
    integrated_label = []
    for dl in integrated_label_lists:
        for kd in dl:
            integrated_label.append(kd)
            
    print("Formating data time {:.3f}s".format(time.time()-formate_time))
    
    print("End data integration")
    print("Stats | Total Integration Time {:.3f}s | totle data number {}| normal number {}| abnormal number {}| label not given number {}"
          .format((time.time()-start_time), len(integrated_label),integrated_label.count(0),integrated_label.count(1),integrated_label.count(-1)))
    
    if save_intergrated_data:
        data_filename = '%s/intergratedlist.json' % (root_processed_data)
        saving_data = []
        
        for itgd in integrated_data:
            saving_data.append(json.loads(itgd.to_json()))
        with open(data_filename, "w") as fp:
            json.dump(saving_data, fp)
            
        label_filename = '%s/listlabel.json' % (root_processed_data)
        with open(label_filename, 'w') as f:
            for l in integrated_label:
                f.write(str(l) + '\n')
                
        print("Save data integrations in folder {}".format(root_processed_data))
    return integrated_data, integrated_label

def thread_integration(kid, dates, k, times, pool_total, all_data, label_data):
    integrated_data = []
    integrated_label = []
    count_entity = 0
    start_time = time.time()
    entity_time = start_time
    for d in dates:
        key_date_label = label_data.loc[(label_data['KEY'] == k) & (label_data['Time'] == d.strftime('%Y/%-m/%-d'))] #d.strftime('%Y/%#m/%#d') for windows
        if not key_date_label.empty:
            key_date_label = key_date_label.reset_index(drop=True)
            key_date_data = pd.DataFrame(columns=all_data.columns)
            for t in times:
                dts = dt.datetime.combine(d,t).strftime('%Y-%m-%d %H:%M:%S')
                key_date_time_data = all_data.loc[(all_data['KEY'] == k) & (all_data['Time'] == dts)]
                if key_date_time_data.empty:
                    key_date_time_data = key_date_time_data.append(pd.Series(dtype='float64'),ignore_index=True)
                    key_date_time_data.loc[0,'KEY'] = k
                    key_date_time_data.loc[0,'Time'] = dts
                key_date_data = key_date_data.append(key_date_time_data, ignore_index=True)
            integrated_data.append(key_date_data)
            integrated_label.append(key_date_label.loc[0,'label'])
        else:
            key_date_data = pd.DataFrame(columns=all_data.columns)
            count_empty = 0
            for t in times:
                dts = dt.datetime.combine(d,t).strftime('%Y-%m-%d %H:%M:%S')
                key_date_time_data = all_data.loc[(all_data['KEY'] == k) & (all_data['Time'] == dts)]
                if key_date_time_data.empty:
                    count_empty += 1
                    key_date_time_data = key_date_time_data.append(pd.Series(dtype='float64'),ignore_index=True)
                    key_date_time_data.loc[0,'KEY'] = k
                    key_date_time_data.loc[0,'Time'] = dts
                key_date_data = key_date_data.append(key_date_time_data, ignore_index=True)
            if count_empty < 96:
                integrated_data.append(key_date_data)
                integrated_label.append(-1)
        
        if d == dates[-1]:
            print("    Pool Integration {}/{}| Ave. Time {:.3f}s | Entity {} {} | Label {}".format(kid, pool_total, (time.time()-entity_time)/(count_entity+1), k, d.strftime('%Y/%m/%d'), integrated_label[-1]))
        count_entity += 1
        entity_time = time.time()
    return integrated_data, integrated_label


