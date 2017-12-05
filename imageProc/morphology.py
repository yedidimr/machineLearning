import numpy as np
import cv2


img_path = "/home/student-5/Documents/morphology/pic1dirty.jpg"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
cv2.imshow('original img', img)


kernel = np.ones((15,15),np.uint8)
disc_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

#clean salt and peper
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, disc_kernel )
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, disc_kernel )

cv2.imshow('slat & peper clean', img)

# clean the boundaries
erosion = cv2.erode(img, kernel, iterations = 1)
cv2.imshow('Erosion - boundaries clean', img)

dilation = cv2.dilate(img, kernel, iterations = 1)
cv2.imshow('Dilation', img)

# connected component labeling in python
ret, thresh = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY_INV)
a = cv2.bitwise_not(thresh)
ret, labels = cv2.connectedComponents(a)
print ret, labels

cv2.imshow('labels', (labels*128).astype('uint8'))
cv2.waitKey(0)


