#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 17:59:39 2021

@author: zichan
"""
import os
import pandas as pd
import numpy as np
import random
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
sns.set_style("white")
import warnings
warnings.filterwarnings("ignore")

def VisAnavisualization(args, visualize_data, out_label, root_image = './visualization', visualize = 'all'):
    
    if not os.path.exists(root_image):
        os.mkdir(root_image)
    
    link_alldates_visualize = False
    links_date_visualize = False
    feature_distribution_visualize = False
    t_sne_visualization = False
    if visualize == 'all':
        link_alldates_visualize = True
        links_date_visualize = True
        feature_distribution_visualize = True
        t_sne_visualization = True
    if visualize == 'preliminary':
        link_alldates_visualize = True
        links_date_visualize = True
    if visualize == 'feature_distribution':
        feature_distribution_visualize = True
    if visualize == 'tsne':
        t_sne_visualization = True
    
    title_str = ''
    file_str = ''
    if args.target_visualize_data == 'corrected':
        title_str = 'Corrected Data'
        file_str = 'corrected'
    
    if args.target_visualize_data in ['transformed', 'visAna_embed']:
        title_str = 'VisAnaML'
        file_str = 'VisAnaML_transformed'
        
    if args.target_visualize_data == 'raw':
        title_str = 'Raw Data'
        file_str = 'raw_integrated'
        
    if args.target_visualize_data in ['pmads', 'pamds_embed']:
        title_str = 'PMADS'
        file_str = 'pmads'
        
    # if args.target_visualize_data == 'visAna_embed':
    #     title_str = 'VisAnaML'
    
    if link_alldates_visualize:
        # step1 a: one line 26-day timeline visualization for preliminary analysis
        print("Start {} target link {} all date time line visualization".format(file_str,args.target_visualize_link))
        visualize_link_alldates(visualize_data, out_label, [title_str, file_str], root_image=root_image,targetLink=args.target_visualize_link)
        
    # step1 b: random(5 normal and 5 abnormal) links at random date time-line features visualization for preliminary analysis
    if links_date_visualize:
        print("Start {} sampled links target date {} time line visualization".format(file_str,args.target_visualize_date))
        visualize_links_date(visualize_data, out_label, [title_str, file_str], root_image=root_image,targetDate=args.target_visualize_date)
    
    # step2 feature distribution visualization
    if feature_distribution_visualize:
        print("Start {} feature distribution visualization".format(file_str))
        visualize_feature_distribution(visualize_data, out_label, [title_str, file_str], root_image=root_image)
    
    # t-sne visualization
    if t_sne_visualization:
        print(1)
    
def visualize_link_alldates(visualize_data, out_label, name_strs, root_image='./visualization', targetLink='AAAI-aa-1---AATF-aa-1'):
    integrated_label = out_label
    visualize_link = pd.DataFrame(columns=visualize_data[0].columns)
    visualize_link_label = []
    for idx,ldf in enumerate(visualize_data):
        ldf.index = pd.RangeIndex(len(ldf.index))
        if ldf.loc[0,'KEY'] == targetLink:
            visualize_link = visualize_link.append(ldf, ignore_index=True)
            visualize_link_label.append(integrated_label[idx])
    visualize_link = visualize_link.sort_values('Time')
    datetimes = visualize_link['Time'].unique()
    datetimes = [dt.datetime.strptime(datetimes[i], '%Y-%m-%d %H:%M:%S') for i in range(len(datetimes))]
    dates = list(set([datetimes[i].date() for i in range(len(datetimes))]))
    dates.sort()
    ## continuous label lists
    continuous_labels_list = []
    label_list = []
    i = 0
    visualize_link_label.append('end')
    while i < len(visualize_link_label)-1:
        previous_label = visualize_link_label[i]
        index_label = visualize_link_label[i]
        while index_label == previous_label:
            label_list.append(i)
            previous_label = index_label
            i += 1
            index_label = visualize_link_label[i]
        continuous_labels_list.append(label_list)
        label_list = []
    visualize_link_label.remove('end')

    color_code = ['darkolivegreen', 'skyblue', 'mediumvioletred']
    
    feature_names = pd.DataFrame(visualize_link.columns)
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
            
    for ft, feature_indexes in enumerate([SNR_index_list, RSL_index_list, EI_index_list]):
        if ft == 0:
            title = '{} {} SNR Features Time-line Visualization\n'.format(name_strs[0], targetLink)
            link_alldate_imagefile = '{}/{}_{}_SNR_alldate.png'.format(root_image,name_strs[1],targetLink)
        if ft == 1:
            title = '{} {} RSL Features Time-line Visualization\n'.format(name_strs[0], targetLink)
            link_alldate_imagefile = '{}/{}_{}_RSL_alldate.png'.format(root_image,name_strs[1],targetLink)
        if ft == 2:
            title = '{} {} Error Indicator Features Time-line Visualization\n'.format(name_strs[0], targetLink)
            link_alldate_imagefile = '{}/{}_{}_EI_alldate.png'.format(root_image,name_strs[1],targetLink)
        fig, axes = plt.subplots(len(feature_indexes), 1, figsize=(50, 60), sharex=True)
        fig.suptitle(title, fontsize=80)
        for fi, f in enumerate(feature_indexes):
            ax = axes[fi]
            col = visualize_link.iloc[:,f]
            for index_label in continuous_labels_list:
                y = pd.Series(name=col.name,dtype='float64')
                for d in index_label:
                    y = y.append(col.loc[d*96:(d+1)*96])
                ax.plot(y,color=color_code[visualize_link_label[index_label[0]]+1],linewidth=6)
            ax.set_ylabel(col.name,fontsize=48)
            if f == feature_indexes[-1]:
                ax.xaxis.set_tick_params(direction='out')
                ax.xaxis.set_ticks_position('bottom')
                ax.set_xlabel('Time',fontsize=74)
                ax.set_xticks(np.arange(48, len(dates)*96-1,96))
                ax.set_xticklabels(dates,fontsize=30,rotation=30)
                ax.set_xlim(0, len(dates)*96-1)
        fig.tight_layout()
        fig.savefig(link_alldate_imagefile)
        fig.clear()
        plt.clf()
        
        
def visualize_links_date(visualize_data, out_label, name_strs, root_image='./visualization',targetDate = '2017/10/8'):
    out_labels = pd.DataFrame(out_label, columns = ['label'])
    links_date_index = []
    for idx, ldf in enumerate(visualize_data):
        ldf.index = pd.RangeIndex(len(ldf.index))
        if dt.datetime.strptime(ldf.loc[0,'Time'], '%Y-%m-%d %H:%M:%S').date() == dt.datetime.strptime(targetDate, '%Y/%m/%d').date():
            links_date_index.append(idx)
    links_date_label = out_labels.loc[links_date_index,'label']
    index_0 = links_date_label.index[links_date_label == 0].tolist()
    index_1 = links_date_label.index[links_date_label == 1].tolist()
    # index_ng = links_date_label.index[links_date_label == -1].tolist()
    # random.shuffle(index_0)
    # random.shuffle(index_1)
    # random.shuffle(index_ng)
    num_link_label = 10
    index_0 = index_0[0:num_link_label]
    index_1 = index_1[0:num_link_label]
    # if len(index_ng) >= 5:
    #     index_ng = index_ng[0:5]
    
    links_date_keys = []
    visualize_links_date = []
    visualize_links_date_label = []
    for index in (index_0+index_1):#+index_ng):
        visualize_data[index].index = pd.RangeIndex(len(visualize_data[index].index))
        visualize_links_date.append(visualize_data[index])
        links_date_keys.append(visualize_data[index].loc[0,'KEY'])
        visualize_links_date_label.append(out_labels.loc[index,'label'])
    
    datetimes = visualize_links_date[0]['Time'].unique()
    datetimes = [dt.datetime.strptime(datetimes[i], '%Y-%m-%d %H:%M:%S') for i in range(len(datetimes))]
    datetimes.sort()
    xtickts_index = [i*8 for i in range(12)]
    xtickts = [dt.datetime.strftime(datetimes[i], '%H:%M') for i in xtickts_index]
    
    # color_code = [['yellowgreen','palegreen','orange','goldenrod','olivedrab'], 
    #               ['skyblue','teal','lightseagreen','deepskyblue','steelblue'], 
    #               ['mediumvioletred', 'crimson', 'deeppink', 'mediumorchid','orchid']]
    color_code = [mpl.cm.get_cmap("GnBu", num_link_label)(np.linspace(0, 1, num_link_label)),
                  mpl.cm.get_cmap("YlOrBr", num_link_label)(np.linspace(0, 1, num_link_label))]
    
    feature_names = pd.DataFrame(visualize_links_date[0].columns)
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

    for ft, feature_indexes in enumerate([SNR_index_list, RSL_index_list, EI_index_list]):
        if ft == 0:
            title = '{} SNR Features {} Time-line Visualization\n'.format(name_strs[0], targetDate)
            link_alldate_imagefile = '{}/{}_SNR_{}.png'.format(root_image,name_strs[1],targetDate.replace("/", ""))
        if ft == 1:
            title = '{} RSL Features {} Time-line Visualization\n'.format(name_strs[0], targetDate)
            link_alldate_imagefile = '{}/{}_RSL_{}.png'.format(root_image,name_strs[1],targetDate.replace("/", ""))
        if ft == 2:
            title = '{} Error Indicator Features {} Time-line Visualization\n'.format(name_strs[0], targetDate)
            link_alldate_imagefile = '{}/{}_EI_{}.png'.format(root_image,name_strs[1],targetDate.replace("/", ""))
        fig, axes = plt.subplots(len(feature_indexes), 2, figsize=(50, 60), sharex=True, sharey='row')
        fig.suptitle(title, fontsize=80)
        for fi, f in enumerate(feature_indexes):
            for li in [0,1]:
                ax = axes[fi][li]
                currentax_indexes = [i for i, label in enumerate(visualize_links_date_label) if label == li]
                for ci, key_idx in enumerate(currentax_indexes):
                    col = visualize_links_date[key_idx].iloc[:,f]
                    ax.plot(col, color=color_code[li][ci], linewidth=6,label=links_date_keys[key_idx])
                    if li == 0 and ci == 0:
                        ax.set_ylabel(col.name,fontsize=50)
                if fi == 0:
                    ax.legend(fancybox=True, framealpha=0.5,prop={'size': 20})
                    # if li == 0:
                    #     ax.set_title('Label Not Given Links\n',fontsize=60)
                    if li == 0:
                        ax.set_title('Normal Links\n',fontsize=60)
                    if li == 1:
                        ax.set_title('Abnormal Links\n',fontsize=60)
                if f == feature_indexes[-1]:
                    ax.xaxis.set_tick_params(direction='out')
                    ax.xaxis.set_ticks_position('bottom')
                    ax.set_xlabel('Time',fontsize=74)
                    ax.set_xticks(np.arange(2, len(datetimes)+1,8))
                    ax.set_xticklabels(xtickts,fontsize=30,rotation=30)
                    ax.set_xlim(0, len(datetimes)-1)
        fig.tight_layout()
        fig.savefig(link_alldate_imagefile)
        fig.clear()
        plt.clf()
        
def visualize_feature_distribution(visualize_data, out_label, name_strs, root_image='./visualization'):
    out_labels = pd.DataFrame(out_label, columns = ['label'])
    index_0 = out_labels.index[out_labels['label'] == 0].tolist()
    index_1 = out_labels.index[out_labels['label'] == 1].tolist()
    # index_ng = out_labels.index[out_labels['label'] == -1].tolist()
    # random.shuffle(index_0)
    # random.shuffle(index_1)
    # random.shuffle(index_ng)
    num_link_label = 100
    index_0 = index_0[0:num_link_label]
    index_1 = index_1[0:num_link_label]
    
    # index_df_ng = []
    # for x in index_ng:
    #     index_df_ng=index_df_ng+list(range(x*96, (x+1)*96))
    
    if not isinstance(visualize_data, pd.DataFrame):
        visualize_feature_dist = pd.DataFrame(columns=visualize_data[0].columns)
        index_df_0 = []
        index_df_1 = []
        for idx, dfi in enumerate(index_0+index_1):
            visualize_feature_dist = visualize_feature_dist.append(visualize_data[dfi], ignore_index=True)
            if dfi in index_0:
                index_df_0=index_df_0+list(range(idx*96, (idx+1)*96))
            if dfi in index_1:
                index_df_1=index_df_1+list(range(idx*96, (idx+1)*96))
    else:
        visualize_feature_dist = visualize_data
        index_df_0 = []
        for x in index_0:
            index_df_0=index_df_0+list(range(x*96, (x+1)*96))
        index_df_1 = []
        for x in index_1:
            index_df_1=index_df_1+list(range(x*96, (x+1)*96))
    
    kwargs = dict(bins=5, norm_hist=True, hist_kws={'alpha':.6}, kde_kws={'linewidth':6})
    color_code = ['skyblue', 'mediumvioletred']#,'darkolivegreen']
    fig, axes = plt.subplots(7, 3, figsize=(50, 60))
    fig.suptitle('{} Feature Distribution\n'.format(name_strs[0]), fontsize=80)
    for axr in range(7):
        for axc in range(3):
            ax = axes[axr][axc]
            f = 3*axr+axc
            for ci, label_idx in enumerate([index_df_0,index_df_1]):#,index_ng]):
                col = visualize_feature_dist.iloc[label_idx,f+2]
                sns.distplot(col , color=color_code[ci], ax=ax, **kwargs)
                ax.set_xlabel(col.name, fontsize=40)
            ax.set_ylabel('Density', fontsize=30)
    fig.tight_layout()
    fig.savefig('{}/{}_feature_dist.png'.format(root_image,name_strs[1]))
    fig.clear()
    plt.clf()