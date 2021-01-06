import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import filters
import cv2
from mpl_toolkits import mplot3d
from collections import defaultdict

from CIELAB_color_space import RGBtoLAB

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

def euclideanEnergy(inputImage):
    img = np.array(inputImage)
    img = RGBtoLAB(img)
    width, height, _ = img.shape

    EUx = np.zeros ([width, height])
    EUy = np.zeros ([width, height])
    result = np.zeros ([width, height])

    for i in range(width-2):
        for j in range(height-2):
            EUx[i, j] = np.linalg.norm(img[i, j+1] - img[i, j])
            EUy[i, j] = np.linalg.norm(img[i+1, j] - img[i, j])

    result = EUx + EUy
    result *= 255.0 / np.max(result)

    # keep value higer than mean
    mask = result > np.mean(np.unique(result))
    result = result * mask

    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(inputImage)
    # plt.subplot(222)
    # plt.imshow(EUx, cmap="gray")
    # plt.subplot(223)
    # plt.imshow(EUy, cmap="gray")
    # plt.subplot(224)
    # plt.imshow(result, cmap="gray")
    # plt.show()

    return result

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

def RGBcolorClassify(img):
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
            
            val_R = int(img[i][j][2] / 51)
            val_G = int(img[i][j][1] / 51)
            val_B = int(img[i][j][0] / 51)
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

def RGBcolorEnergy(img, do_blur):
    """
    Color classes with highest frequency indicates lowest energy, vice versa.
    """
    _, color_importance = RGBcolorClassify(img)
    
    img_new = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            # 可以跟 colorClassify 合併加速
            val_R = int(img[i][j][2] / 51)
            val_G = int(img[i][j][1] / 51)
            val_B = int(img[i][j][0] / 51)
            class_num = str(val_R) + str(val_G) + str(val_B)
            
            img_new[i][j] = color_importance[class_num]
    
    blur_img = filters.gaussian_filter(img_new, sigma=1)

    if do_blur == True:
        return blur_img
    else:
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

def LABcolorEnergy(img, do_blur):
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
    
    blur_img = filters.gaussian_filter(img_new, sigma=1)

    if do_blur == True:
        return blur_img
    else:
        return img_new

def combineEnergy(Edge_Eng, color_Eng, do_gamma):
    gamma = 0.4
    gamma_trans = np.array(255*(Edge_Eng / 255) ** gamma, dtype = 'uint8')

    if do_gamma == True:
        combine_Eng = gamma_trans + color_Eng
    else:
        combine_Eng = Edge_Eng + color_Eng

    max_val = np.max(combine_Eng)
    combine_Eng = combine_Eng * (255/max_val)

    return combine_Eng

if __name__ == "__main__":

    img = cv2.imread('./image/dolphin.jpg')
    resultDir = 'dolphin'

    do_blur = True
    do_gamma = True
    
    if (do_blur == True) and (do_gamma == True):
        resultDir = resultDir + '_blur_gamma_result/'

    elif (do_blur == True) and (do_gamma == False):
        resultDir = resultDir + '_blur_result/'

    elif (do_blur == False) and (do_gamma == True):
        resultDir = resultDir + '_gamma_result/'

    elif (do_blur == False) and (do_gamma == False):
        resultDir = resultDir + '_result/'

    if not os.path.exists(resultDir):
        os.mkdir(resultDir)


    plt.figure('RGB color space')
    plt.subplot(221)
    plt.title("RGB Sobel Energy")
    sobel_Eng = SobelEnergy(img, 'RGB')
    plt.imshow(sobel_Eng, cmap="gray")
    cv2.imwrite(resultDir+'RGBsobel_Eng.jpg', sobel_Eng)

    plt.subplot(222)
    plt.title("RGB Color Energy")
    RGBcolor_Eng = RGBcolorEnergy(img, do_blur)
    plt.imshow(RGBcolor_Eng, cmap="gray")
    cv2.imwrite(resultDir+'RGBcolor_Eng.jpg', RGBcolor_Eng)

    plt.subplot(223)
    plt.title("RGB Combined Energy")
    combine_Eng = combineEnergy(sobel_Eng, RGBcolor_Eng, do_gamma)
    plt.imshow(combine_Eng, cmap="gray")
    cv2.imwrite(resultDir+'RGBcombine_Eng.jpg', combine_Eng)

    plt.figure('LAB color space')
    plt.subplot(221)
    plt.title("LAB Sobel Energy")
    LABeuclidean_Eng = euclideanEnergy(img)
    plt.imshow(LABeuclidean_Eng, cmap="gray")
    cv2.imwrite(resultDir+'LABeuclidean_Eng.jpg', LABeuclidean_Eng)
    
    plt.subplot(222)
    plt.title("LAB Color Energy")
    LABcolor_Eng = LABcolorEnergy(img, do_blur)
    plt.imshow(LABcolor_Eng, cmap="gray")
    cv2.imwrite(resultDir+'LABcolor_Eng.jpg', LABcolor_Eng)

    plt.subplot(223)
    plt.title("LAB Combined Energy")
    LABcombine_Eng = combineEnergy(LABeuclidean_Eng, LABcolor_Eng, do_gamma)
    plt.imshow(LABcombine_Eng, cmap="gray")
    cv2.imwrite(resultDir+'LABcombine_Eng.jpg', LABcombine_Eng)
    plt.show()
