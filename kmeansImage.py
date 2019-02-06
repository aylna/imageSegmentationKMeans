#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 22:39:50 2019

@author: aylin
"""

import cv2 
import numpy as np
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 

im = cv2.imread("a1.jpg") 
im = im[::, ::, ::-1] # BGR -> RGB 

im = cv2.GaussianBlur(im, (27, 27), 3) 

h = im.shape[0] # height
w = im.shape[1] # width
xx, yy = np.meshgrid(np.arange(w), np.arange(h)) # coordinate matrix
xx = np.expand_dims(xx, axis=2) # h * w * 1 array


yy = np.expand_dims(yy, axis=2)

data = np.concatenate((im, xx, yy), axis=2) # arrays to h * w * 5 
# print data.shape 

# normalization (x - mean) / std
data_ = np.reshape(data, [-1, data.shape[-1]])
data_ = (data_ - data_.mean(axis=0)) / data_.std(axis=0)


data_[::, -1] *= 0.3
data_[::, -2] *= 0.3


cluster_sizes = [3, 4, 5]

plt.subplot(1, 1+len(cluster_sizes), 1)
plt.imshow(im)
for i, cs in enumerate(cluster_sizes):
    cls = KMeans(n_clusters=cs)
    res = cls.fit_predict(data_)
    res = np.reshape(res, [data.shape[0], data.shape[1]])

    plt.subplot(1, 1 + len(cluster_sizes), i+2 )
    plt.imshow(res, cmap="Spectral")
plt.show()