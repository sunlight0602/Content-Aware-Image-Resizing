#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 17:14:50 2020

Seam carving main process
with feed forward
*Takes time*

Reference: https://zhuanlan.zhihu.com/p/38974520

Adjust parameters in line31~34
    color_div: number of divisions of color space
    img_name: file name of target image
    do_blur: conduct gaussian bluring on energy map
    do_gamma: conduct gamma transform on color ranking
"""

import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from Energy import rgbSobelEnergy
from Energy import labSobelEnergy
from Energy import RGBcolorEnergy
from Energy import LABcolorEnergy
from Energy import combineEnergy

color_div = 52
img_name = 'dolphin.jpg'
do_blur = False
do_gamma = True


def minimum_seam(img, map_type):
    r, c, _ = img.shape

    if map_type=='RGBsobel':
        energy_map = rgbSobelEnergy(img, 'RGB')
    elif map_type=='RGBcolor':
        energy_map = RGBcolorEnergy(img, color_div, do_blur)
    elif map_type=='RGBcombine':
        energy_map = combineEnergy(rgbSobelEnergy(img, 'RGB'), RGBcolorEnergy(img,color_div,do_blur), do_gamma)
    elif map_type=='LABsobel':
        energy_map = labSobelEnergy(img)
    elif map_type=='LABcolor':
        energy_map = LABcolorEnergy(img, color_div, do_blur)
    elif map_type=='LABcombine':
        energy_map = combineEnergy(labSobelEnergy(img), LABcolorEnergy(img,color_div,do_blur), do_gamma)
    elif map_type=='RGBsobelLABcolorcombine':
        energy_map = combineEnergy(rgbSobelEnergy(img, 'RGB'), LABcolorEnergy(img,color_div,do_blur), do_gamma)

    # Show carving process
    # plt.imshow(energy_map, cmap="gray")
    # plt.show()
    
    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
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


def carve_column(img, map_type):
    """
    Carve one seam.
    """
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img, map_type)

    mask = np.ones((r, c), dtype=np.bool)

    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis=2)

    img = img[mask].reshape((r, c-1, 3))

    return img


def crop_c(img, scale_c, map_type):
    """
    Calculate new img size.
    """
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in trange(c - new_c):
        img = carve_column(img, map_type)

    return img


if __name__=='__main__':
    
    # Create destiny file
    resultDir = img_name[:-4]+'_result/'
    if not os.path.exists(resultDir):
        os.mkdir(resultDir)

    # Read image
    img = cv2.imread('./image/'+img_name)
    saveName = 'Edge_enhance_'
    if img is None:
        sys.exit("no img")
    
    # Carve and show
    # RGB sobel
    print("\nDoing RGB Sobel Energy...")
    plt.subplot(231)
    # plt.title("RGB Sobel Energy")
    rgbsobel = crop_c(img, 0.8, 'RGBsobel')
    rgbsobel = rgbsobel.astype(np.uint8)
    plt.imshow(cv2.cvtColor(rgbsobel, cv2.COLOR_BGR2RGB))
    fileName = resultDir + 'rgbsobel' + '.jpg'
    cv2.imwrite(fileName, rgbsobel)

    # RGB color
    print("\nDoing RGB Color Energy...")
    plt.subplot(232)
    # plt.title("RGB Color Energy")
    rgbcolor = crop_c(img, 0.8, 'RGBcolor')
    plt.imshow(cv2.cvtColor(rgbcolor, cv2.COLOR_BGR2RGB))
    fileName = resultDir + 'rgbcolor_' + str(color_div) + '.jpg'
    cv2.imwrite(fileName, rgbcolor)

    # RGB sobel + color
    print("\nDoing RGB Combine Energy...")
    plt.subplot(233)
    # plt.title("RGB Combine Energy")
    rgbcombine = crop_c(img, 0.8, 'RGBcombine')
    plt.imshow(cv2.cvtColor(rgbcombine, cv2.COLOR_BGR2RGB))
    fileName = resultDir + 'rgbcombine_' + str(color_div) + '.jpg'
    cv2.imwrite(fileName, rgbcombine)

    # LAB sobel
    print("\nDoing LAB Sobel Energy...")
    plt.subplot(234)
    # plt.title("LAB Sobel Energy")
    labsobel = crop_c(img, 0.8, 'LABsobel')
    plt.imshow(cv2.cvtColor(labsobel, cv2.COLOR_BGR2RGB))
    fileName = resultDir + 'labsobel' + '.jpg'
    cv2.imwrite(fileName, labsobel)

    # LAB color
    print("\nDoing LAB Color Energy...")
    plt.subplot(235)
    # plt.title("LAB color Energy")
    labcolor = crop_c(img, 0.8, 'LABcolor')
    plt.imshow(cv2.cvtColor(labcolor, cv2.COLOR_BGR2RGB))
    fileName = resultDir + 'labcolor_' + str(color_div) + '.jpg'
    cv2.imwrite(fileName, labcolor)

    # LAB sobel + color
    print("\nDoing LAB Combine Energy...")
    plt.subplot(236)
    # plt.title("LAB Combine Energy")
    labcombine = crop_c(img, 0.8, 'LABcombine')
    plt.imshow(cv2.cvtColor(labcombine, cv2.COLOR_BGR2RGB))
    fileName = resultDir + 'labcombine_' + str(color_div) + '.jpg'
    cv2.imwrite(fileName, labcombine)
    
    plt.show()
    
    # RGB sobel + LAB color
    print("\nDoing RGB sobel + LAB color combine Energy...")
    combine = crop_c(img, 0.8, 'RGBsobelLABcolorcombine')
    plt.imshow(cv2.cvtColor(combine, cv2.COLOR_BGR2RGB))
    fileName = resultDir + 'combine_' + str(color_div) + '.jpg'
    cv2.imwrite(fileName, combine)
