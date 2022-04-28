#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 02:29:10 2021

@author: zichan
"""

import os
visualization = False
visual_folder = 'D:/zichanr/RoAd/log/forvis'

dataset_name = 'VisAnaML'

net_name = 'VisAnaML_LSTM'

device = 'cuda:7'

gammaes = [1]
# gammaes = [1,2]

radiuses = [8]
# radiuses = [1,2,4,6,8,10,12,16]

ascent_step_sizees = [0.001]
# ascent_step_sizees = [0.001,0.01,0.1]

ascent_num_stepses = [100]
# ascent_num_stepses = [100,10]

ratio_unlabels = [0]
# ratio_unlabels = [0,0.2,0.5,0.8]

n_pollutions = [0]
# n_pollutions = [0,100,200,400,800]

mislabel_types = ['agnostic']
# mislabel_types = ['agnostic', 'asym_o_n', 'asym_n_o']

mislabel_ratios = [0, 0.2, 0.5, 0.8]
# mislabel_ratio = [0.2, 0.5, 0.8]

wotranss = [1]
# wotranss = [0,1]

woadvs = [0]
# woadvs = [0,1]

seeds = [1,2,3,4,5]
# seeds = [1,2,3,4,5]

iters1, iters2, iters3, iters4, iters5, iters6, iters7 = ascent_num_stepses, ascent_step_sizees, wotranss, woadvs, gammaes, mislabel_ratios, seeds
a = []
for i in range(len(iters1)):
    a += [iters1[i]]*len(iters2)*len(iters3)*len(iters4)*len(iters5)*len(iters6)*len(iters7)
b = []
for i in range(len(iters2)):
    b += [iters2[i]]*len(iters3)*len(iters4)*len(iters5)*len(iters6)*len(iters7)
b = b * len(iters1)
c = []
for i in range(len(iters3)):
    c += [iters3[i]]*len(iters4)*len(iters5)*len(iters6)*len(iters7)
c = c*len(iters2)*len(iters1)
d = []
for i in range(len(iters4)):
    d += [iters4[i]]*len(iters5)*len(iters6)*len(iters7)
d = d *len(iters3)*len(iters2)*len(iters1)
e = []
for i in range(len(iters5)):
    e += [iters5[i]]*len(iters6)*len(iters7)
e = e *len(iters4)*len(iters3)*len(iters2)*len(iters1)
f = []
for i in range(len(iters6)):
    f += [iters6[i]]*len(iters7)
f = f *len(iters5)*len(iters4)*len(iters3)*len(iters2)*len(iters1)
g = iters7 *len(iters6)*len(iters5)*len(iters4)*len(iters3)*len(iters2)*len(iters1)

if not visualization:
    for iter1, iter2, iter3, iter4, iter5, iter6, iter7 in zip(a,b,c,d,e,f,g):
        xp_path = '_'.join(mislabel_types, str(iter6))
        os.system("python  ./main.py   \
                    --xp_path D:/zichanr/RoAd/log/{}   \
                    --data_path D:/zichanr/RoAd/data     \
                    --load_model_from_pretrain D:/zichanr/RoAd/log/pretrainmodel_wofpre/model.tar \
                    --pretrain 0 \
                    --radius {} \
                    --gamma {} \
                    --ascent_num_steps {} \
                    --ascent_step_size {} \
                    --n_pollution {}   \
                    --ratio_unlabel {}     \
                    --mislabel_type {}  \
                    --mislabel_ratio {}  \
                    --lr 0.0001   \
                    --n_epochs 150   \
                    --lr_milestone 50 100    \
                    --batch_size 128  \
                    --weight_decay 0.5e-6     \
                    --ae_lr 0.0001    \
                    --ae_n_epochs 150   \
                    --ae_lr_milestone 80     \
                    --ae_batch_size 128   \
                    --ae_weight_decay 0.5e-3  \
                    --wotrans {} \
                    --woadv {} \
                    --device {}   \
                    --seed {}   \
                    --exp_str {} ".format(xp_path,radiuses[0],iter5,iter1,iter2,n_pollutions[0],ratio_unlabels[0],
                    mislabel_types[0],iter6,iter3,iter4,device,iter7,iter7))    
                    # .format(radius,gamma,ascent_num_steps,ascent_step_size,n_pollution,ratio_unlable,
                    # mislabel_type,mislabel_ratio,wotrans,woadv,device,seed,exp_str))            
                    # --ae_n_epochs 150 --ae_lr 0.0001 --ae_n_epochs 150 --ae_weight_decay 0.5e-3 --ae_lr_milestone 50
              
## python main.py --xp_path D:/zichanr/RoAd/log/ --data_path D:/zichanr/RoAd/data --load_model_from_pretrain D:/zichanr/RoAd/log/pretrain_combineae/model.tar --pretrain 0 --mislabel_ratio 0 --wotrans 1 --woadv 1 --device cuda:7 --exp_str modeltesting        
else:
    for file in os.listdir(visual_folder):
        os.system("python  ./main.py   \
                    --xp_path D:/zichanr/RoAd/log/visresults   \
                    --data_path D:/zichanr/RoAd/data     \
                    --load_model {}/{}/model.tar \
                    --pretrain 0 \
                    --visualization 1   \
                    --device {}   \
                    --exp_str {} ".format(visual_folder, file, device, file.split('_')[-1]))
    