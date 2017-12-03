import cv2
import numpy as np
from scipy import signal

img_path = "/home/student-5/Downloads/large-im2.jpg"

def quant(img, gray_level):
    assert gray_level <= 8 and gray_level >=0
    assert type(gray_level) is int
    scale = 256/(2**gray_level)
    img = img/scale  # downscale
    return img * scale  # upscale

img = cv2.imread(img_path, 0)  # read as frayscale

cv2.imshow('original - 8 levels' , img)
img4 = quant(img,2)
cv2.imshow('down to 4 levels' , img4)
# gaussian_1d =  signal.gaussian(min(img.shape), 1).reshape((min(img.shape),1))#, sym=True)
# print gaussian_1d.shape
# gaussian_2d = np.dot(gaussian_1d, np.transpose(gaussian_1d))
# np.pad
# print img.shape, gaussian_2d.shape
# gaussian_addition = np.zeros(img.shape)
# gaussian_addition [:gaussian_2d.shape[0],:gaussian_2d.shape[1]] = gaussian_2d
# print np.amin(gaussian_addition), np.amax(gaussian_addition)
# img += gaussian_addition
# cv2.imshow('gaussian' , img)
cv2.waitKey(0)
