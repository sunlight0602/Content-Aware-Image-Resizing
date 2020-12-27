import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import filters


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
    
    plotProcess(imx, imy, result)

    return result

def plotProcess(imx, imy, energy):
    plt.subplot(131)
    plt.title("imx")
    plt.imshow(imx, cmap="gray")

    plt.subplot(132)
    plt.title("imy")
    plt.imshow(imy, cmap="gray")

    plt.subplot(133)
    plt.title("energy")
    plt.imshow(energy, cmap="gray")

    plt.show()

    

if __name__ == "__main__":
    pli_img = Image.open("./image/castle.jpg")
    
    sobelEng = SobelEnergy(pli_img)

    