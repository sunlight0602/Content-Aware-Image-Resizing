#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 17:14:50 2020

Seam carving main process
*Takes time*

Reference: https://zhuanlan.zhihu.com/p/38974520
"""

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image

from Energy import SobelEnergy
from Energy import RGBcolorEnergy
from Energy import LABcolorEnergy
from Energy import combineEnergy

def minimum_seam(img, energy_map):
    r, c, _ = img.shape

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            # 处理图像的左侧边缘，确保我们不会索引-1
            if j == 0:
                idx = np.argmin( M[i-1, j:j+2] )
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx+j]
            else:
                idx = np.argmin( M[i-1, j-1:j+2] )
                backtrack[i, j] = idx + j - 1
                min_energy = M[i-1, idx+j-1]

            M[i, j] += min_energy

    return M, backtrack

def carve_column(img, energy_map):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img, energy_map)

    # 创建一个(r,c)矩阵，填充值为True
    # 后面会从值为False的图像中移除所有像素
    mask = np.ones((r, c), dtype=np.bool)

    # 找到M的最后一行中的最小元素的位置 
    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        # 标记出需要删除的像素
        mask[i, j] = False
        j = backtrack[i, j]

    # 因为图像有3个通道，我们将蒙版转换为3D
    mask = np.stack([mask] * 3, axis=2)

    # 删除蒙版中所有标记为False的像素，
    # 将照片及能量圖大小重新调整为新图像的维度
    img = img[mask].reshape((r, c-1, 3))
    energy_map = energy_map[mask[:, :, 0]].reshape((r, c-1))

    return img, energy_map

def crop_c(img, scale_c, map_type):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    if map_type=='RGBsobel':
        energy_map = SobelEnergy(img, 'RGB')
    elif map_type=='RGBcolor':
        energy_map = RGBcolorEnergy(img)
    elif map_type=='RGBcombine':
        energy_map = combineEnergy(SobelEnergy(img, 'RGB'), LABcolorEnergy(img))
    elif map_type=='LABsobel':
        energy_map = SobelEnergy(img, 'LAB')
    elif map_type=='LABcolor':
        energy_map = LABcolorEnergy(img)
    elif map_type=='LABcombine':
        energy_map = combineEnergy(SobelEnergy(img, 'LAB'), LABcolorEnergy(img))

    r_eng, c_eng = energy_map.shape

    for i in trange(c - new_c): # use range if you don't want to use tqdm
        img, energy_map = carve_column(img, energy_map)

    return img

if __name__=='__main__':
    
    # Read image
    img = cv2.imread('./image/dolphin.jpg')
    saveName = 'Edge_enhance_'
    if img is None:
        sys.exit("no img")
    
    # Show orig image
    # plt.subplot(221)
    # plt.title("Original")
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.show()
    
    # Carve and show
    print("Doing RGBSobel Energy...")
    plt.subplot(231)
    plt.title("RGB Sobel Energy")
    rgbsobel = crop_c(img, 0.8, 'RGBsobel')
    rgbsobel = rgbsobel.astype(np.uint8)
    print(rgbsobel.dtype) 
    plt.imshow(cv2.cvtColor(rgbsobel, cv2.COLOR_BGR2RGB))
    fileName = saveName + 'rgbsobel' + '.jpg'
    cv2.imwrite(fileName, rgbsobel)

    print("Doing RGBColor Energy...")
    plt.subplot(232)
    plt.title("RGB Color Energy")
    rgbcolor = crop_c(img, 0.8, 'RGBcolor')
    plt.imshow(cv2.cvtColor(rgbcolor, cv2.COLOR_BGR2RGB))
    fileName = saveName + 'rgbcolor' + '.jpg'
    cv2.imwrite(fileName, rgbcolor)

    print("Doing RGB Combine Energy...")
    plt.subplot(233)
    plt.title("RGB Combine Energy")
    rgbcombine = crop_c(img, 0.8, 'RGBcombine')
    plt.imshow(cv2.cvtColor(rgbcombine, cv2.COLOR_BGR2RGB))
    fileName = saveName + 'rgbcombine' + '.jpg'
    cv2.imwrite(fileName, rgbcombine)

    print("Doing LABSobel Energy...")
    plt.subplot(234)
    plt.title("LAB Sobel Energy")
    labsobel = crop_c(img, 0.8, 'LABsobel')
    plt.imshow(cv2.cvtColor(labsobel, cv2.COLOR_BGR2RGB))
    fileName = saveName + 'labsobel' + '.jpg'
    cv2.imwrite(fileName, labsobel)


    print("Doing LAB Color Energy...")
    plt.subplot(235)
    plt.title("LAB color Energy")
    labcolor = crop_c(img, 0.8, 'LABcolor')
    plt.imshow(cv2.cvtColor(labcolor, cv2.COLOR_BGR2RGB))
    fileName = saveName + 'labcolor' + '.jpg'
    cv2.imwrite(fileName, labcolor)

    print("Doing LAB Combine Energy...")
    plt.subplot(236)
    plt.title("LAB Combine Energy")
    labcombine = crop_c(img, 0.8, 'LABcombine')
    plt.imshow(cv2.cvtColor(labcombine, cv2.COLOR_BGR2RGB))
    fileName = saveName + 'labcombine' + '.jpg'
    cv2.imwrite(fileName, labcombine)
    
    plt.show()
