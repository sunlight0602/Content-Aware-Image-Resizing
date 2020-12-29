import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import filters
import cv2
import math

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
    if t>0.008856:
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

if __name__ == "__main__":
    #pli_img = Image.open("./image/pika.png")
    #sobelEng = SobelEnergy(pli_img)
    
    img = cv2.imread('./image/pika.png')
    img_LAB = RGBtoLAB(img)
    
