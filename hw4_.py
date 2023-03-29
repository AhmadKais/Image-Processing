import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.filters import median_filter


def clean_baby(im):
    h,w=im.shape
    clean_im = signal.medfilt(im,5)
    clean_im=clean_im[22:128,6:111]
    clean_im=cv2.resize(clean_im, (w,h), interpolation = cv2.INTER_AREA)
    return clean_im


#id did not get the same result that azmi got check that before submition
def clean_windmill(im):
    IM = np.fft.fft2(im)
    IMs = np.fft.fftshift(IM)
    N = IMs.shape[0]
    x, y = np.meshgrid(np.arange(N), np.arange(N))
    a1 = 0.008
    a2 = 0.008
    NF1 = 1 - np.exp(-a1 * (x - 157) ** 2 - a2 * (y - 131) ** 2)  # Gaussian
    NF2 = 1 - np.exp(-a1 * (x - 100) ** 2 - a2 * (y - 124) ** 2)  # Gaussian
    Z = NF1 * NF2
    IMFs = IMs * Z
    IMFr = np.fft.ifftshift(IMFs)
    imfr = np.fft.ifft2(IMFr)
    clean_im = np.real(imfr)
    clean_im = np.real(clean_im)
    return clean_im


def clean_watermelon(im):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    clean_im = im
    for i in range(2):
        clean_im = cv2.filter2D(src=clean_im, ddepth=-1, kernel=kernel)
    return clean_im


def clean_umbrella(im):
    fft_image = np.fft.fft2(im)
    clean_im = fft_image
    n,m = im.shape
    mk = np.zeros([n, m])
    mk[0][0] = 0.5
    #if you try different cordinatoes it wont work we need to go back over the tutorials
    mk[4][79] = 0.5
    mk = np.fft.fft2(mk)
    mk[abs(mk) < 0.01] = 1
    clean_im = clean_im/mk
    clean_im = abs(np.fft.ifft2(clean_im))
    return clean_im

def clean_USAflag(im):
    temp_im = im.copy()
    temp_im= median_filter(temp_im, [1,70])
    temp_im[0:85, 0:170] = im[0:85, 0:170]
    clean_im = temp_im
    return clean_im


def clean_cups(im):
    invGamma = 1.0 / 1.5
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    clean_im=cv2.LUT(im, table)
    clean_im = cv2.LUT(clean_im, table)
    return clean_im

def clean_house(im):
    fft_im = np.fft.fft2(im)
    n,m  = im.shape
    cleanIm= np.zeros([n,m])
    cleanIm[0][0:10] = 0.1
    cleanIm= np.fft.fft2(cleanIm)
    cleanIm[abs(cleanIm) < 0.01] = 1
    cleanIm = fft_im /cleanIm
    return abs(np.fft.ifft2(cleanIm))

# use histogram equalization we did this in previous hw
def clean_bears(im):
    hist = cv2.calcHist(im, [0], None, [256], [0, 256])
    gray_img_eqhist = cv2.equalizeHist(im)
    return gray_img_eqhist


'''
    # an example of how to use fourier transform:
    img = cv2.imread(r'Images\windmill.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_fourier = np.fft.fft2(img) # fft - remember this is a complex numbers matrix 
    img_fourier = np.fft.fftshift(img_fourier) # shift so that the DC is in the middle

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title('original image')

    plt.subplot(1,3,2)
    plt.imshow(np.log(abs(img_fourier)), cmap='gray') # need to use abs because it is complex, the log is just so that we can see the difference in values with out eyes.
    plt.title('fourier transform of image')

    img_inv = np.fft.ifft2(img_fourier)
    plt.subplot(1,3,3)
    plt.imshow(abs(img_inv), cmap='gray')
    plt.title('inverse fourier of the fourier transform')

'''




