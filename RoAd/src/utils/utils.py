#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:16:23 2021

@author: zichan
"""

import torch
import shutil
import os
import numpy as np
from sklearn.metrics import confusion_matrix, pairwise_distances
# from sklearn.manifold import TSNE
from sklearn.utils.multiclass import unique_labels
import tensorflow as tf
def prepare_folders(args):
    """Create log and model folder"""
    folders_util = [args.xp_path,
                    os.path.join(args.xp_path, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)
    
def binary_crossentropy(y_true, y_pred, sample_weight=1):
    epsilon = 1e-07
    if len(y_pred.shape)==1:
        y_pred = np.atleast_2d(y_pred).T
    y_pred = [max(min(pred[0], 1-epsilon), epsilon) for pred in y_pred]
    y_true,y_pred,sample_weight = force_2d_shape([y_true,y_pred,sample_weight])

    logits = np.log(y_pred) - np.log(1-y_pred) # sigmoid inverse
    neg_abs_logits = -np.abs(logits)
    relu_logits = (logits > 0)*logits
    
    loss_vec = relu_logits - logits*y_true + np.log(1 + np.exp(neg_abs_logits))
    return torch.tensor(np.mean(sample_weight*loss_vec))

def force_2d_shape(arr_list):
    for arr_idx, arr in enumerate(arr_list):
        if len(np.array(arr).shape) != 2:
            arr_list[arr_idx] = np.atleast_2d(arr).T
    return arr_list