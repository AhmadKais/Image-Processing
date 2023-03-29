import numpy as np
import matplotlib.pyplot as plt
import cv2

def histImage(im):
    h = np.empty(256)
    h.fill(0)
    for x in range(256):
        val = np.sum(im==x)
        h[x] = val
    return h

def nhistImage(im):
    nh = np.empty(256)
    nh.fill(0)
    hm = histImage(im)
    sumim=sum(hm)
    for x in range(256):
        nh[x] = hm[x]/sumim
    return nh

def ahistImage(im):
    vector = np.empty(256)
    vector.fill(0)
    hist = histImage(im)
    val = 0
    vector[0] = val
    i = 0
    for x in range(256):
        vector[x] = val
        val = val + hist[i]
        i+=1
    ah = vector
    ah.reshape(256, 1)
    return ah


def calcHistStat(h):
    m = np.matmul([i for i in range(256)], h) / (sum(h))
    e = sum((np.power([i for i in range(256)] - m, 2) * h)) / (sum(h))
    return m, e


def mapImage(im, tm):
    a,b = im.shape
    nim = im
    finalimage = np.full((a, b), 0)
    for x in range(256):
        tempimage = nim.copy()
        tempimage = np.where(tempimage == x,tm[x],0)
        finalimage = finalimage+tempimage
    finalimage[finalimage > 255] = 255
    finalimage[finalimage < 0] = 0
    return finalimage


def histEqualization(im):
    acchist = ahistImage(im)
    tm = np.empty(256)
    eqhis = np.empty(256)
    eqhis.fill(sum(histImage(im))/ 256)
    gmhist = np.cumsum(eqhis)
    i = 0
    for x in range(256):
        while gmhist[i] < acchist[x]:
            i += 1
        tm[x] = i
    tm = tm.reshape(256, 1)
    return tm
