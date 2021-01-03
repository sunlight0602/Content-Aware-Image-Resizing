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

from Energy import SobelEnergy
from Energy import colorEnergy
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

    if map_type=='sobel':
        energy_map = SobelEnergy(img)
    elif map_type=='color':
        energy_map = colorEnergy(img)
    elif map_type=='LABcolor':
        energy_map = LABcolorEnergy(img)
    elif map_type=='combine':
        energy_map = combineEnergy(SobelEnergy(img), LABcolorEnergy(img))

    r_eng, c_eng = energy_map.shape

    for i in trange(c - new_c): # use range if you don't want to use tqdm
        img, energy_map = carve_column(img, energy_map)

    return img

if __name__=='__main__':
    
    # Read image
    img = cv2.imread('./image/peak.jpg')
    if img is None:
        sys.exit("no img")
    
    # Show orig image
    plt.subplot(221)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.show()
    
    # Carve and show
    # print("Doing Sobel Energy...")
    # plt.subplot(222)
    # plt.title("Sobel Energy")
    # c_img1 = crop_c(img, 0.5, 'sobel')
    # plt.imshow(cv2.cvtColor(c_img1, cv2.COLOR_BGR2RGB))

    print("Doing LAB Color Energy...")
    plt.subplot(222)
    plt.title("LAB color Energy")
    c_img1 = crop_c(img, 0.8, 'LABcolor')
    plt.imshow(cv2.cvtColor(c_img1, cv2.COLOR_BGR2RGB))
    
    print("Doing Color Energy...")
    plt.subplot(223)
    plt.title("Color Energy")
    c_img2 = crop_c(img, 0.8, 'color')
    plt.imshow(cv2.cvtColor(c_img2, cv2.COLOR_BGR2RGB))

    print("Doing Combine Energy...")
    plt.subplot(224)
    plt.title("Combine Energy")
    c_img3 = crop_c(img, 0.8, 'combine')
    plt.imshow(cv2.cvtColor(c_img3, cv2.COLOR_BGR2RGB))

    plt.show()
