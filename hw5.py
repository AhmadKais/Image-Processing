
import numpy as np
from scipy.signal import convolve2d as conv
from scipy.ndimage.filters import gaussian_filter as gaussian
import matplotlib.pyplot as plt
import cv2

def sobel(im):
    saf = 120
    sobel_aofki = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_amody = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    edges_x = conv(im, sobel_aofki, mode="same", boundary="symm")
    edges_y = conv(im, sobel_amody, mode="same", boundary="symm")
    edge_len = np.sqrt(edges_x ** 2 + edges_y ** 2)
    edge_len[edge_len < saf] = 0
    edge_len[edge_len >= saf] = 1
    new_im = edge_len
    return new_im.astype(np.uint8)

def canny(im):
    blur = cv2.GaussianBlur(im, (9, 9), 0)
    threshold = 60
    edge_image = cv2.Canny(blur, threshold, threshold+190)
    return edge_image

def hough_circles(im):
    new_im = im.copy()
    another_im = im.copy()
    new_im = cv2.medianBlur(new_im, 9)
    cimg = new_im
    circles = cv2.HoughCircles(new_im, cv2.HOUGH_GRADIENT, 1, 20,param1=120, param2=40, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(another_im, (i[0], i[1]), i[2], (0, 255, 0), 3)
        cv2.circle(another_im, (i[0], i[1]), 1, (0, 0, 255), 3)
    return another_im

def hough_lines(im):
    new_im = im.copy()
    cannt_trns = cv2.Canny(new_im, 235, 255, apertureSize=3)
    lines = cv2.HoughLines(cannt_trns, 1, np.pi / 60, 150)
    for line in lines:
        var, theta = line[0]
        b = np.sin(theta)
        a = np.cos(theta)
        x0 = a * var
        y0 = b * var
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(new_im, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return new_im
