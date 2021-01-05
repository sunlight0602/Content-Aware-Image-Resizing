import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import filters
import cv2
from mpl_toolkits import mplot3d
from collections import defaultdict

from CIELAB_color_space import RGBtoLAB

RGB_color_div = 50
img_name = 'pika4.jpg'

def SobelEnergy(inputImage, colorType):
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

def plotProcess(imx, imy, energy):
    plt.subplot(221)
    plt.title("imx")
    plt.imshow(imx, cmap="gray")

    plt.subplot(222)
    plt.title("imy")
    plt.imshow(imy, cmap="gray")

    plt.subplot(223)
    plt.title("energy")
    plt.imshow(energy, cmap="gray")
    plt.show()

    return

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

def RGBcolorClassify(img, div):
    """
    Parameters
    ----------
    img : input image, 3d in RGB color space

    Returns
    -------
    color_dict : dict
        key has 3 digits, representing R,G,B value respectively
        digits range from 0~5, calculated by floor(value/51)
        
        value is the number of pixels of the color class
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
    # total_pixel = sum(color_freq.values())
    # importance = [ [k, v/total_pixel] for k,v in color_freq.items() ]
    # importance.sort(key = lambda x: x[1])
    # keys = [ e[0] for e in importance ]
    # values = [ e[1] for e in importance ]
    # values.reverse()
    
    # importance = [ [keys[i], values[i]] for i in range(len(keys)) ]
    
    # num = 255/importance[0][1]
    # for i, _ in enumerate(importance):
    #     importance[i][1] = importance[i][1] * num
        
    # color_importance = defaultdict(int)
    # for ele in importance:
    #     color_importance[ele[0]] = ele[1]
    
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

def RGBcolorEnergy(img, div):
    """
    Color classes with highest frequency indicates lowest energy, vice versa.
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
    
    return img_new

def LABcolorClassify(img):
    """
    Parameters
    ----------
    img : input image, 3d in LAB color space

    Returns
    -------
    color_dict : dict
        Key has 3 digits, representing L,a,b value respectively.
        Devide L into 10 slots, calculated by floor(value/10) (L range 0~100).
        Devide a into 5 slots, calculated by floor(value+128/51) (L range -128~127).
        Devide b into 5 slots, calculated by floor(value+128/51) (L range -128~127).
        
        value is the number of pixels of the color class
    """
    
    # Classify
    color_freq = defaultdict(int) # 0,1,2,3,4,5
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            val_L = int(img[i][j][0] / 25)
            val_a = int(img[i][j][1]+128 / 51)
            val_b = int(img[i][j][2]+128 / 51)
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

def LABcolorEnergy(img):
    """
    Color classes with highest frequency indicates lowest energy, vice versa.
    """
    LABimg = RGBtoLAB(img)
    _, color_importance = LABcolorClassify(LABimg)
    
    img_new = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            # 可以跟 LABcolorClassify 合併加速
            val_L = int(LABimg[i][j][0] / 25)
            val_a = int(LABimg[i][j][1]+128 / 51)
            val_b = int(LABimg[i][j][2]+128 / 51)
            class_num = str(val_L) + str(val_a) + str(val_b)
            
            img_new[i][j] = color_importance[class_num]
    
    return img_new

def combineEnergy(sobel_Eng, color_Eng):
    combine_Eng = sobel_Eng*1.3 + color_Eng
    max_val = np.max(combine_Eng)
    combine_Eng = combine_Eng * (255/max_val)

    return combine_Eng

if __name__ == "__main__":

    resultDir = img_name[:-4]+'_result/'
    if not os.path.exists(resultDir):
        os.mkdir(resultDir)

    img = cv2.imread('./image/'+img_name)

    plt.subplot(231)
    # plt.title("RGB Sobel Energy")
    sobel_Eng = SobelEnergy(img, 'RGB')
    plt.imshow(sobel_Eng, cmap="gray")
    cv2.imwrite(resultDir+'RGBsobel_Eng.jpg', sobel_Eng)

    plt.subplot(232)
    # plt.title("RGB Color Energy")
    RGBcolor_Eng = RGBcolorEnergy(img, RGB_color_div)
    plt.imshow(RGBcolor_Eng, cmap="gray")
    cv2.imwrite(resultDir+'RGBcolor_Eng_'+ str(RGB_color_div) +'.jpg', RGBcolor_Eng)

    plt.subplot(233)
    # plt.title("RGB Combined Energy")
    combine_Eng = combineEnergy(sobel_Eng, RGBcolor_Eng)
    plt.imshow(combine_Eng, cmap="gray")
    cv2.imwrite(resultDir+'RGBcombine_Eng_'+ str(RGB_color_div) +'.jpg', combine_Eng)

    plt.subplot(234)
    # plt.title("LAB Sobel Energy")
    LABsobel_Eng = SobelEnergy(img, 'LAB')
    plt.imshow(LABsobel_Eng, cmap="gray")
    cv2.imwrite(resultDir+'LABsobel_Eng.jpg', LABsobel_Eng)
    
    plt.subplot(235)
    # plt.title("LAB Color Energy")
    LABcolor_Eng = LABcolorEnergy(img)
    plt.imshow(LABcolor_Eng, cmap="gray")
    cv2.imwrite(resultDir+'LABcolor_Eng.jpg', LABcolor_Eng)

    plt.subplot(236)
    # plt.title("LAB Combined Energy")
    LABcombine_Eng = combineEnergy(sobel_Eng, LABcolor_Eng)
    plt.imshow(LABcombine_Eng, cmap="gray")
    cv2.imwrite(resultDir+'LABcombine_Eng.jpg', LABcombine_Eng)
    plt.show()

    # img = cv2.imread('./image/pika.png')
    # RGBHistogram(img)
    
    # plt.subplot(224)
    # plt.title("LAB image")
    # img_LAB = RGBtoLAB(img)
    # print(np.max(img_LAB[:, :, 0]))
    # plt.imshow(img_LAB[:, :, 0])

    # img = cv2.imread('./image/castle.jpg')
    # plotRGBSpace(img)
    
    # img = cv2.imread('./image/pika2.png')
    # color_freq, color_importance = colorClassify(img)
