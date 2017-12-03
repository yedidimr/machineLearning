import cv2
import numpy as np

kernel_1d = [0.05,  0.25, 0.4, 0.25, 0.05]
KERNEL = np.outer(kernel_1d, kernel_1d)

img_path1 = "/home/student-5/Downloads/im1.jpg"
img_path2 = "/home/student-5/Downloads/im2.jpg"
mask_path = "/home/student-5/Downloads/msk.jpg"

im1 = cv2.imread(img_path1, 0)
im2 = cv2.imread(img_path2, 0)
msk = cv2.imread(mask_path, 0)

# print im1.shape, im2.shape, msk.shape  #
# p1= msk[i]*im1[i]
# p2=(1 - msk[i])*im2[i]
# build new imgp1+p2

# https://compvisionlab.wordpress.com/2013/05/13/image-blending-using-pyramid/
def expand(img, factor=2):
    new_shape = (img.shape[1]*factor, img.shape[0]*factor)
    new = cv2.resize(img, new_shape, interpolation=cv2.INTER_NEAREST)
    new = cv2.filter2D(new, -1, KERNEL)  # convolve2d
    return new

def reduce(img, factor=2):
    new = cv2.filter2D(img, -1, KERNEL)  # convolve2d
    return new[::factor, ::factor]

def genGaussianPyramid(img, num_levels):
    last_img = img
    pyr = [img]
    for i in range(num_levels):
        last_img = reduce(last_img)
        pyr.append(last_img)
    return pyr


def genLaplacianPyramid(gaussian_pyr):
    # returns number of levels
    expanded = [expand(pyr) for pyr in gaussian_pyr]
    return [gaussian_pyr[i] - expanded[i] for i in range(len(gaussian_pyr))]


cv2.imshow('original', img)
img_s = expand(img, 2)
img_b = reduce(img, 2)
cv2.imshow('expanded', img_s)
cv2.imshow('reduce', img_b)

cv2.waitKey(0)