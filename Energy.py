"""
Functions for drawing structure-detection maps and color-ranking maps.

Adjust parameters on line19-22 for code testing.

color_div : number of divisions to divide color space.
img_name : filename of input image.
do_blur : do gaussian-blurring on color-ranking maps.
do_gamma : do gamma-transform on structure-detection maps.
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import filters
import cv2
from collections import defaultdict

from CIELAB_color_space import RGBtoLAB

color_div = 52
img_name = 'dolphin.jpg'
do_blur = False
do_gamma = True


def rgbSobelEnergy(inputImage, colorType='RGB'):
    """Draw structure-detection map using Sobel-filtering.
    Parameters
    ----------
    inputImage : input image.
    colorType : set to 'LAB' if inputImage is in LAB color space. Defaults to 'RGB'.

    Returns
    -------
    result : structure-detection map using Sobel-filtering.
    """
    img = np.array(inputImage)

    if colorType == 'LAB':
        img = RGBtoLAB(img)

    imR = img[:, :, 0]
    imG = img[:, :, 1]
    imB = img[:, :, 2]

    Rx = np.zeros (imR.shape)
    Gx = np.zeros (imG.shape)
    Bx = np.zeros (imB.shape)

    Ry = np.zeros (imR.shape)
    Gy = np.zeros (imG.shape)
    By = np.zeros (imB.shape)

    # 1 for horizontal derivative
    filters.sobel(imG, 1, Gx)
    filters.sobel(imR, 1, Rx)
    filters.sobel(imB, 1, Bx)

    imx = np.zeros (imB.shape)
    imx = Rx + Gx + Bx
    imx *= 255.0 / np.max(imx)


    # 0 for vertical derivative
    filters.sobel(imR, 0, Ry)
    filters.sobel(imG, 0, Gy)
    filters.sobel(imB, 0, By)

    imy = np.zeros (imB.shape)
    imy = Ry + Gy + By
    imy *= 255.0 / np.max(imy)

    # combine imx and imy
    result = np.hypot(imx, imy)  # magnitude
    result *= 255.0 / np.max(result) # normalize
    
    # uncomment to plot sobel proccess
    # plotProcess(imx, imy, result)

    return result


def labSobelEnergy(inputImage):
    """Draw structure-detection map using Euclidean distance.
    Parameters
    ----------
    inputImage : input image.

    Returns
    -------
    result : structure-detection map using Euclidean distance.
    """
    img = np.array(inputImage)
    img = RGBtoLAB(img)
    width, height, _ = img.shape

    EUx = np.zeros([width, height])
    EUy = np.zeros([width, height])
    result = np.zeros([width, height])

    for i in range(width-2):
        for j in range(height-2):
            EUx[i, j] = np.linalg.norm(img[i, j+1] - img[i, j])
            EUy[i, j] = np.linalg.norm(img[i+1, j] - img[i, j])

    result = EUx + EUy
    result *= 255.0 / np.max(result)
    
    # keep value higer than mean, (denoise)
    mask = result > np.mean(np.unique(result))
    result = result * mask

    return result


def RGBcolorClassify(img, div):
    """
    Parameters
    ----------
    img : input image, 3 tunnels in RGB color space.
    div : number of divisions to be divided.

    Returns
    -------
    color_dict : dict
        Key has 3 digits, representing R,G,B value respectively.
        Digits range from 0 ~ int(255/div).
        Value is the number of pixels of the division.

    color_importance : dict
        color_dict with order.
    """
    
    # Classify
    color_freq = defaultdict(int) # 0,1,2,3,4,5
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            val_R = int(img[i][j][2] / div)
            val_G = int(img[i][j][1] / div)
            val_B = int(img[i][j][0] / div)
            class_num = str(val_R) + str(val_G) + str(val_B)
            
            color_freq[class_num] += 1
            

    # Importance    
    total_pixel = sum(color_freq.values())

    importance = [ [k, v/total_pixel] for k,v in color_freq.items() ]

    importance.sort(key = lambda x: x[1])
    keys = [ e[0] for e in importance ]
    values = [ e[1] for e in importance ]
    
    importance = [ [keys[i], len(keys)-i] for i in range(len(keys)) ]

    num = 255/importance[0][1]
    for i, _ in enumerate(importance):
        importance[i][1] = importance[i][1] * num
    
    color_importance = defaultdict(int)
    for ele in importance:
        color_importance[ele[0]] = ele[1]
    
    return color_freq, color_importance


def RGBcolorEnergy(img, div, do_blur):
    """Obtain color-ranking map based on RGB color space.
    Parameters
    ----------
    img : original image.
    div : number of divisions to be divided.
    do_blur (bool) : conduct gaussian-filtering on output map.

    Returns
    -------
    blur_img : color-ranking map with gaussian-filtering.
    or
    img_new : color-ranking map without gaussian-filtering.
    """
    _, color_importance = RGBcolorClassify(img, div)
    
    img_new = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            # 可以跟 colorClassify 合併加速
            val_R = int(img[i][j][2] / div)
            val_G = int(img[i][j][1] / div)
            val_B = int(img[i][j][0] / div)
            class_num = str(val_R) + str(val_G) + str(val_B)
            
            img_new[i][j] = color_importance[class_num]

    blur_img = filters.gaussian_filter(img_new, sigma=1)

    if do_blur == True:
        return blur_img
    else:
        return img_new


def LABcolorClassify(img, div):
    """Divide LAB color space and give rankings to each division.
    Parameters
    ----------
    img : input image, 3d in LAB color space.
    div : number of divisions to be divided.

    Returns
    -------
    color_dict : dict
        Key has 3 digits, representing L,a,b value respectively.
        Divide L into 10 slots, calculated by floor(value/10) (L range 0~100).
        Divide a into 5 slots, calculated by floor(value+128/51) (L range -128~127).
        Divide b into 5 slots, calculated by floor(value+128/51) (L range -128~127).
        value is the number of pixels of the color class.
        
    color_importance : dict
        color_dict with order.
    """
    
    # Classify
    color_freq = defaultdict(int) # 0,1,2,3,4,5
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            val_L = int(img[i][j][0] / 25)
            val_a = int(img[i][j][1]+128 / div)
            val_b = int(img[i][j][2]+128 / div)
            class_num = str(val_L) + str(val_a) + str(val_b)
            
            color_freq[class_num] += 1
    
    total_pixel = sum(color_freq.values())

    importance = [ [k, v/total_pixel] for k,v in color_freq.items() ]

    importance.sort(key = lambda x: x[1])
    keys = [ e[0] for e in importance ]
    values = [ e[1] for e in importance ]
    
    importance = [ [keys[i], len(keys)-i] for i in range(len(keys)) ]

    num = 255/importance[0][1]
    for i, _ in enumerate(importance):
        importance[i][1] = importance[i][1] * num
    
    color_importance = defaultdict(int)
    for ele in importance:
        color_importance[ele[0]] = ele[1]
    
    return color_freq, color_importance


def LABcolorEnergy(img, div, do_blur):
    """Obtain color-ranking map based on LAB color space.
    Parameters
    ----------
    img : original image.
    div : number of divisions to be divided.
    do_blur (bool) : conduct gaussian-filtering on output map.

    Returns
    -------
    blur_img : color-ranking map with gaussian-filtering.
    or
    img_new : color-ranking map without gaussian-filtering.
    """
    
    LABimg = RGBtoLAB(img)
    _, color_importance = LABcolorClassify(LABimg, div)
    
    img_new = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            # 可以跟 LABcolorClassify 合併加速
            val_L = int(LABimg[i][j][0] / 25)
            val_a = int(LABimg[i][j][1]+128 / div)
            val_b = int(LABimg[i][j][2]+128 / div)
            class_num = str(val_L) + str(val_a) + str(val_b)
            
            img_new[i][j] = color_importance[class_num]
    
    blur_img = filters.gaussian_filter(img_new, sigma=1)

    if do_blur == True:
        return blur_img
    else:
        return img_new


def combineEnergy(edge_Eng, color_Eng, do_gamma):
    """Combine structure-detection map and color-ranking map
    Parameters
    ----------
    edge_Eng : structure-detection map.
    color_Eng : color-ranking map.
    do_gamma (bool) : conduct gamma transformation on structure-detection map.

    Returns
    -------
    combine_Eng : combined map.
    """
    
    if do_gamma == True:
        gamma = 0.4
        gamma_trans = np.array(255*(edge_Eng / 255) ** gamma, dtype = 'uint8')
        combine_Eng = gamma_trans + color_Eng
    else:
        combine_Eng = edge_Eng + color_Eng
    
    max_val = np.max(combine_Eng)
    combine_Eng = combine_Eng * (255/max_val)

    return combine_Eng


if __name__ == "__main__":

    resultDir = img_name[:-4]+'_result/'
    if not os.path.exists(resultDir):
        os.mkdir(resultDir)

    img = cv2.imread('./image/'+img_name)

    plt.figure()
    # plt.subplot(231)
    plt.title("RGB sobel")
    RGBsobel_Eng = rgbSobelEnergy(img, 'RGB')
    plt.imshow(RGBsobel_Eng, cmap="gray")
    cv2.imwrite(resultDir+'RGBsobel_Eng.jpg', RGBsobel_Eng)

    plt.figure()
    # plt.subplot(232)
    plt.title("RGB color")
    RGBcolor_Eng = RGBcolorEnergy(img, color_div, do_blur)
    plt.imshow(RGBcolor_Eng, cmap="gray")
    cv2.imwrite(resultDir+'RGBcolor_Eng_'+ str(color_div) +'.jpg', RGBcolor_Eng)

    plt.figure()
    # plt.subplot(233)
    plt.title("RGB sobel + color")
    combine_Eng = combineEnergy(RGBsobel_Eng, RGBcolor_Eng, do_gamma)
    plt.imshow(combine_Eng, cmap="gray")
    cv2.imwrite(resultDir+'RGBcombine_Eng_'+ str(color_div) +'.jpg', combine_Eng)

    plt.figure()
    # plt.subplot(234)
    plt.title("LAB sobel")
    LABsobel_Eng = labSobelEnergy(img)
    plt.imshow(LABsobel_Eng, cmap="gray")
    cv2.imwrite(resultDir+'LABsobel_Eng.jpg', LABsobel_Eng)
    
    plt.figure()
    # plt.subplot(235)
    plt.title("LAB color")
    LABcolor_Eng = LABcolorEnergy(img, color_div, do_blur)
    plt.imshow(LABcolor_Eng, cmap="gray")
    cv2.imwrite(resultDir+'LABcolor_Eng_'+str(color_div)+'.jpg', LABcolor_Eng)

    plt.figure()
    # plt.subplot(236)
    plt.title("LAB sobel + color")
    LABcombine_Eng = combineEnergy(LABsobel_Eng, LABcolor_Eng, do_gamma)
    plt.imshow(LABcombine_Eng, cmap="gray")
    cv2.imwrite(resultDir+'LABcombine_Eng_'+str(color_div)+'.jpg', LABcombine_Eng)
    plt.show()
    
    #==============euclidean distance=========

    plt.figure()
    plt.title("RGB sobel + LAB color")
    RGBsobel_Eng = rgbSobelEnergy(img, 'RGB')
    LABcolor_Eng = LABcolorEnergy(img, color_div, do_blur)
    combine_Eng = combineEnergy(RGBsobel_Eng, LABcolor_Eng, do_gamma)
    plt.imshow(combine_Eng, cmap="gray")
    cv2.imwrite(resultDir+'combine_Eng_'+str(color_div)+'.jpg', combine_Eng)
    plt.show()
    
    
