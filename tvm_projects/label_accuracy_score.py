# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 17:20:21 2021

@author: Administrator
"""

from skimage.io import imread, imsave
import numpy as np
import os
import gdal

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def label(img_path):
    img = imread(img_path)
    # print(img.shape)
    width = img.shape[0]
    height = img.shape[1]
    new_image = np.random.randint(0, 256, size=[width, height], dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if all(np.array(img[i][j]) == np.array([255, 0, 0])):  # 红色，建成区
                new_image[i][j] = 1
            elif all(np.array(img[i][j]) == np.array([0, 255, 0])):  # 绿色，农用地
                new_image[i][j] = 2
            elif all(np.array(img[i][j]) == np.array([0, 255, 255])):  # 天蓝色，林地
                new_image[i][j] = 3
            elif all(np.array(img[i][j]) == np.array([255, 255, 0])):  # 黄色，草地
                new_image[i][j] = 4
            elif all(np.array(img[i][j]) == np.array([0, 0, 255])):  # 蓝色，水系
                new_image[i][j] = 5
            else:
                new_image[i][j] = 0
            pass
    return new_image

images = os.listdir("../dataset/image_RGB")
images = images[120:]

label_width = 7200
label_height = 6800
n_class = 6

mean_iu_tvm = 0
mean_iu_torch = 0

acc_tvm = 0
acc_torch = 0

acc_cls_tvm = 0
acc_cls_torch = 0

fwavacc_tvm = 0
fwavacc_torch = 0
'''
for i in range(30):
    img_array = label("../test_images_1/gid_test_{}.tif".format(i))
    dataset = gdal.Open("../dataset/label_5classes_train/"+images[i][:-4]+"_label.tif")
    label_array = dataset.ReadAsArray()
    acc, acc_cls, mean_iu, fwavacc = label_accuracy_score([label_array],[img_array],n_class)
    print(i)
    print("tvm:", acc, acc_cls, mean_iu, fwavacc)
    
    mean_iu_tvm = mean_iu_tvm + mean_iu
    acc_tvm = acc_tvm + acc
    acc_cls_tvm = acc_cls_tvm + acc_cls
    fwavacc_tvm = fwavacc_tvm + fwavacc
    
    img_array = label("../test_images_1/gid_test_torch_{}.tif".format(i))
    acc, acc_cls, mean_iu, fwavacc = label_accuracy_score([label_array],[img_array],n_class)
    print("torch:", acc, acc_cls, mean_iu, fwavacc)
    
    mean_iu_torch = mean_iu_torch + mean_iu
    acc_torch = acc_torch + acc
    acc_cls_torch = acc_cls_torch + acc_cls
    fwavacc_torch = fwavacc_torch + fwavacc
    
print(mean_iu_tvm, mean_iu_torch)
print(acc_tvm, acc_torch)
print(acc_cls_tvm, acc_cls_torch)
print(fwavacc_tvm, fwavacc_torch)
'''

for i in range(30):
    img_array = label("../test_images_4/gid_test_resize_{}.tif".format(i))
    dataset = gdal.Open("../dataset/label_5classes_train/"+images[i][:-4]+"_label.tif")
    label_array = dataset.ReadAsArray()
    acc, acc_cls, mean_iu, fwavacc = label_accuracy_score([label_array],[img_array],n_class)
    print(i)
    print(acc, acc_cls, mean_iu, fwavacc)
    
    mean_iu_tvm = mean_iu_tvm + mean_iu
    acc_tvm = acc_tvm + acc
    acc_cls_tvm = acc_cls_tvm + acc_cls
    fwavacc_tvm = fwavacc_tvm + fwavacc
    
print(mean_iu_tvm)
print(acc_tvm)
print(acc_cls_tvm)
print(fwavacc_tvm)
