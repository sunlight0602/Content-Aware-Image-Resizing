#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Support functions
"""

from matplotlib import pyplot as plt
import cv2

def RGBHistogram(img):
    """
    Plot histogram of RGB values.
    """
    plt.figure('RGB Histogram')
    color = ('b','g','r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0, 256])
        plt.plot(histr, color = col)
        plt.xlim([0, 256])
    plt.show()
        
    return

def plotRGBSpace(img):
    """
    Plot 3d RGB color space.
    """
    
    ax = plt.axes(projection="3d")
    
    R_points = [ img[i][j][2] for i in range(img.shape[0]) for j in range(img.shape[1]) ]
    G_points = [ img[i][j][1] for i in range(img.shape[0]) for j in range(img.shape[1]) ]
    B_points = [ img[i][j][0] for i in range(img.shape[0]) for j in range(img.shape[1]) ]
    
    ax.scatter3D(R_points, G_points, B_points, color='r')
    ax.scatter3D(255, 255, 255, c='white')
    ax.scatter3D(0, 0, 0, c='black')
    ax.scatter3D(255, 255, 0, c='yellow')
    ax.scatter3D(0, 0, 255, c='blue')
    ax.scatter3D(255, 0, 0, c='red')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    return

def plotProcess(imx, imy, energy):
    
    plt.title("imx")
    plt.imshow(imx, cmap="gray")

    # plt.subplot(222)
    # plt.title("imy")
    # plt.imshow(imy, cmap="gray")

    # plt.subplot(223)
    # plt.title("energy")
    # plt.imshow(energy, cmap="gray")
    # plt.show()

    return