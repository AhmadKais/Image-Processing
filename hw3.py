import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def add_SP_noise(im, p):
    sp_noise_im = im.copy()
    LofIm = im.shape[0] * im.shape[1]
    perce = p*(LofIm)
    perce = int(perce)
    RS = random.sample(range(0,LofIm),perce)
    sp_noise_im = np.ravel(sp_noise_im)
    sp_noise_im[RS[0::2]] = 0 #jump 2 steps starting from 0
    sp_noise_im[RS[1::2]] = 255 # jump 2 steps starting from 0
    sp_noise_im = np.clip(sp_noise_im, 0, 255)
    sp_noise_im = np.uint8(np.reshape(sp_noise_im, im.shape))
    return sp_noise_im

def clean_SP_noise_single(im, radius):
    noise_im = im.copy()
    n,m = noise_im.shape
    for i in range(radius ,n-radius):
        for j in range(radius , m-radius):
            if noise_im[i, j] == 255 or noise_im[i, j] == 0:
                noise_im[i, j] = np.median(noise_im[max(0, i - radius):min(n, i + radius + 1), max(0, j - radius):min(m, j + radius + 1)])
    clean_im = noise_im
    return clean_im

def clean_SP_noise_multiple(images):
    clean_image = np.median(images, axis=0)
    return clean_image


def add_Gaussian_Noise(im, s):
    gaussian_noise_im = im.copy()
    Gnoise = np.random.normal(0, s, gaussian_noise_im.shape)
    gaussian_noise_im = gaussian_noise_im + Gnoise
    gaussian_noise_im = np.clip(gaussian_noise_im, 0, 255)
    return gaussian_noise_im


def clean_Gaussian_noise(im, radius, maskSTD):
    x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    mask = np.exp(-((x ** 2) + (y ** 2)) / (2 * (maskSTD ** 2)))
    mask = mask.astype(float)
    mask = mask / np.sum(mask)
    im = im.astype(float)
    cleanIm = convolve2d(im, mask, mode='same')
    return cleanIm.astype(np.uint8)

def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    X, Y = np.meshgrid(range(-radius, radius+1), range(-radius, radius+1))
    clean_im = np.zeros_like(im)
    for i in range(radius, im.shape[0]-radius):
        for j in range(radius, im.shape[1]-radius):
            window = im[i-radius:i+radius+1, j-radius:j+radius+1]
            gi = np.exp(-(window - im[i, j])**2 / (2 * stdIntensity**2))
            gs = np.exp(-(((X)**2) + ((Y)**2)) / (2 * stdSpatial**2))
            gi /= gi.sum()
            gs /= gs.sum()
            clean_im = clean_im.astype(float)
            clean_im[i, j] = (gi * gs * window).sum() / (gi * gs).sum()
    return clean_im.astype(np.uint8)




