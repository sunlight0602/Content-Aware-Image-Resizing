import numpy as np
import math
from matplotlib import pyplot as plt
import cv2

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

if __name__ == "__main__":

    img = cv2.imread('./image/dolphin.jpg')

    plt.figure('RGB color space')
    plt.title("RGB Sobel Energy")

    img_Lab = RGBtoLAB(img)
    plt.imshow(img_Lab)
    plt.show()

    cv2.imwrite('img_Lab.jpg', img_Lab)