#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 00:46:39 2020

@author: zichan
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import pickle
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

def sigmoid(x):
    return 1/(1+math.exp(-x))

def _cross_entropy(t,h):
    return -t*math.log(h)-(1-t)*math.log(1-h)
 
def BCE(y_true, y_pred):
    bce = 0
    m = np.shape(y_pred)[1]*np.shape(y_pred)[0]
    for i in range(np.shape(y_pred)[0]):
        for j in range(np.shape(y_pred)[1]):
            t = y_true[i][j]
            # h = sigmoid(input[i][j])
            h = y_pred[i][j]
            #print(h)
            bce += _cross_entropy(t,h)
    bce /= m
    return bce

f1 = open('trainwCallbackHistoryDict.txt', 'rb')
history = pickle.load(f1)

def plot_picture(history):
    '''
    画出训练集和验证集的损失和精度变化，分析模型状态
    :return:
    '''

    # 画出训练集和验证集的损失和精度变化，分析模型状态

    pre = history['precision']  # 训练集pre
    val_pre = history['val_precision']  # 验证集pre
    f1 = history['f1_score']
    val_f1 = history['val_f1_score']
    loss = history['loss']  # 训练损失
    val_loss = history['val_loss']  # 验证损失
    epochs = range(1, len(pre) + 1)  # 迭代次数
    plt.plot(epochs, loss, 'b--', label='Training loss')  # bo for blue dot 蓝色点
    plt.plot(epochs, val_loss, 'r--', label='Validation loss')
    plt.plot(epochs, pre, 'b+-', label='Training pre', markersize=4)  # bo for blue dot 蓝色点
    plt.plot(epochs, val_pre, 'r+-', label='Validation pre', markersize=4)
    plt.plot(epochs, f1, 'b.-', label='Training f1', markersize=4)  # bo for blue dot 蓝色点
    plt.plot(epochs, val_f1, 'r.-', label='Validation f1', markersize=4)
    plt.title('Training and validation loss, precision and f1-score')
    plt.xlabel('Epochs')
    plt.ylabel('Loss and precision')
    plt.legend(loc='best',fontsize='x-small')
    plt.show()