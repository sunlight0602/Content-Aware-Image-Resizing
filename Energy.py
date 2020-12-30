import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import filters
import cv2
import math
from mpl_toolkits import mplot3d
from collections import defaultdict

def SobelEnergy(inputImage):
    
    img = np.array(inputImage)
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
    
    #plotProcess(imx, imy, result)

    return result

def plotProcess(imx, imy, energy):
    #plt.subplot(131)
    plt.title("imx")
    plt.imshow(imx, cmap="gray")
    plt.show()

    #plt.subplot(132)
    plt.title("imy")
    plt.imshow(imy, cmap="gray")
    plt.show()

    #plt.subplot(133)
    plt.title("energy")
    plt.imshow(energy, cmap="gray")
    plt.show()

    return
    
def RGBtoLAB(img):
    """
    Turn RGB color space to CIELAB space.
    """
    img = img/255.0
    
    CIE_XYZ = [[0.412453, 0.357580, 0.180423],\
               [0.212671, 0.715160, 0.072169],\
               [0.019334, 0.119193, 0.950227]]
    
    imgR = img[:,:,0]
    imgG = img[:,:,1]
    imgB = img[:,:,2]
    
    img_XYZ = np.zeros( (img.shape[0], img.shape[1], 3) )

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            R = imgR[i][j]
            G = imgG[i][j]
            B = imgB[i][j]
            img_XYZ[i][j] = np.dot(CIE_XYZ, [R,G,B])
    
    img_LAB = np.zeros( (img.shape[0], img.shape[1], 3) )
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            X = img_XYZ[i][j][0]
            Y = img_XYZ[i][j][1]
            Z = img_XYZ[i][j][2]
            
            Xn = 0.9515
            Yn = 1.0
            Zn = 1.0886
            
            a = 500 * ( LAB_func(X/Xn) - LAB_func(Y/Yn) )
            b = 200 * ( LAB_func(Y/Yn) - LAB_func(Z/Zn) )
            L = L_func(Y, Yn)
            
            img_LAB[i][j] = [L, a, b]
    
    return img_LAB

def LAB_func(t):
    if t > 0.008856:
        t = math.pow(t, 1/3)
    else:
        t = 7.787 * t + (16/116)
    
    return t

def L_func(Y, Yn):
    if Y/Yn > 0.008856:
        L = 116 * math.pow(Y/Yn, 1/3) - 16
    else:
        L = 903.3 * (Y/Yn)
    
    return L

def RGBHistogram(img):
    """
    Plot histogram of RGB values.
    """

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

def colorClassify(img):
    """
    Parameters
    ----------
    img : input image, 3d in RGB color space

    Returns
    -------
    color_dict : dict
        key has 3 digits, representing R,G,B value respectively
        digits range from 0~5, calculated by floor(value/51)
        
        value is the # of pixels of the color class
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

def colorEnergy(img):
    """
    Color classes with highest frequency indicates lowest energy, vice versa.
    """
    _, color_importance = colorClassify(img)
    
    img_new = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            val_R = int(img[i][j][2] / 51)
            val_G = int(img[i][j][1] / 51)
            val_B = int(img[i][j][0] / 51)
            class_num = str(val_R) + str(val_G) + str(val_B)
            
            img_new[i][j] = color_importance[class_num]
    
    return img_new

if __name__ == "__main__":
    pli_img = Image.open("./image/pika2.png")
    sobelEng = SobelEnergy(pli_img)
    plt.imshow(sobelEng, cmap="gray")
    plt.show()
    
    #img = cv2.imread('./image/pika.png')
    #RGBHistogram(img)
    
    #img = cv2.imread('./image/pika.png')
    #img_LAB = RGBtoLAB(img)
    
    img = cv2.imread('./image/pika2.png')
    plotRGBSpace(img)
    
    img = cv2.imread('./image/pika2.png')
    color_freq, color_importance = colorClassify(img)
    
    img = cv2.imread('./image/pika2.png')
    color_energy = colorEnergy(img)
    plt.imshow(color_energy, cmap="gray")
    plt.show()
