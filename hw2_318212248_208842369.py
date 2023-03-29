import cv2
import matplotlib.pyplot as plt
import numpy as np
# size of the image
m,n = 921, 750

# frame points of the blank wormhole image
src_points = np.float32([[0, 0],
                            [int(n / 3), 0],
                            [int(2 * n /3), 0],
                            [n, 0],
                            [n, m],
                            [int(2 * n / 3), m],
                            [int(n / 3), m],
                            [0, m]])

# blank wormhole frame points
dst_points = np.float32([[96, 282],
                       [220, 276],
                       [344, 276],
                       [468, 282],
                       [474, 710],
                       [350, 744],
                       [227, 742],
                       [103, 714]]
                      )


def find_transform(pointset1, pointset2):
    ps1 = np.copy(pointset1)
    ps2 = np.copy(pointset2)
    newX = np.zeros([16, 8])
    j=0
    for i in range(0,15,2):
        newX[i,0] = ps1[j,0]
        newX[i,1] = ps1[j,1]
        newX[i,2] = 0
        newX[i,3] = 0
        newX[i,5] = 0
        newX[i,4] = 1
        newX[i,6] = -(ps1[j,0])*(ps2[j,0])
        newX[i,7] = -(ps1[j,1])*(ps2[j,0])
        #now fill the odd rows
        newX[i+1, 0] = 0
        newX[i+1, 1] = 0
        newX[i+1, 2] = ps1[j, 0]
        newX[i+1, 3] = ps1[j, 1]
        newX[i+1, 5] = 1
        newX[i+1, 4] = 0
        newX[i+1, 6] = -(ps1[j, 0]) * (ps2[j, 1])
        newX[i+1, 7] = -(ps1[j, 1]) * (ps2[j, 1])
        j = j+1
    invnewX = np.linalg.pinv(newX)
    ps2 = ps2.reshape(16,)
    varMat = np.matmul(invnewX,ps2) # vector  of 8 numbers
    T = np.zeros([3, 3])
    T[0,0] = varMat[0]
    T[0,1] = varMat[1]
    T[0,2] = varMat[4]
    T[1,0] = varMat[2]
    T[1,1] = varMat[3]
    T[1,2] = varMat[5]
    T[2,0] = varMat[6]
    T[2,1] = varMat[7]
    T[2,2] = 1
    return T


def trasnform_image(image, T):
    row,col = image.shape
    new_image = np.zeros([row,col])
    invT = np.linalg.inv(T)
    newvec = np.zeros([3,])
    for x in range(col):
        for y in range(row):
            newvec[0] = x
            newvec[1] = y
            newvec[2] = 1
            newcor = np.matmul(T,newvec)
            xx = newcor[0]
            yy = newcor[1]
            new_image[round(yy),round(xx)] = image[y,x]
    return new_image


def create_wormhole(im, T, iter=5):
    row, col = im.shape
    new_image = np.zeros([row,col])
    new_image = new_image + im
    imtemp = im.copy()
    for i in range(iter):
        ni = trasnform_image(imtemp,T)
        new_image = new_image+ni
        imtemp = ni
    np.clip(new_image, 0, 255)
    return new_image