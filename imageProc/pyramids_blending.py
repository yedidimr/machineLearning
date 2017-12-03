import cv2
import numpy as np

kernel_1d = [0.05,  0.25, 0.4, 0.25, 0.05]
KERNEL = np.outer(kernel_1d, kernel_1d)

img_path1 = "/home/student-5/Downloads/im1.jpg"
img_path2 = "/home/student-5/Downloads/im2.jpg"
mask_path = "/home/student-5/Downloads/msk.jpg"

im1 = cv2.imread(img_path1, 0)
im2 = cv2.imread(img_path2, 0)
# im2 = 0*im2
msk = cv2.imread(mask_path, 0)

print im1.shape, im2.shape, msk.shape
LEVEL = 6

# https://compvisionlab.wordpress.com/2013/05/13/image-blending-using-pyramid/
def expand(img, factor=2):
    new_shape = (img.shape[1]*factor, img.shape[0]*factor)
    new = cv2.resize(img, new_shape)#, interpolation=cv2.INTER_NEAREST)
    new = cv2.filter2D(new, -1, KERNEL)  # convolve2d
    return new
    # return cv2.pyrUp(img)


def reduce(img, factor=2):
    new = cv2.filter2D(img, -1, KERNEL)  # convolve2d
    return new[::factor, ::factor]
    # return cv2.pyrDown(img)


def genGaussianPyramid(img, num_levels):
    last_img = img
    pyr = [img]
    for i in range(num_levels-1):
        last_img = reduce(last_img)
        pyr.append(last_img)
    return pyr

    # G = img.copy()
    # gpM = [G]
    # for i in xrange(LEVEL):
    #     G = cv2.pyrDown(G)
    #     gpM.append(G)
    # print "GAUS", len(gpM),  [ i.shape for i in gpM]
    # return gpM



def genLaplacianPyramid(gaussian_pyr, num_levels):
    # assert num_levels == len(gaussian_pyr) -1
    print "gaus", [ i.shape for i in gaussian_pyr]
    expanded = [expand(pyr) for pyr in gaussian_pyr]
    expanded = expanded[1:] # the first img is double size than the original
    # print "expanded",len(expanded),  [ i.shape for i in expanded]
    # print "gaussian_pyr",len(gaussian_pyr),  [ i.shape for i in gaussian_pyr]
    lap = [gaussian_pyr[t] - expanded[t] for t in range(len(expanded))]  # delta  (diff)
    lap.append(gaussian_pyr[-1])  # last is the original smallest img (and not the diff)

    # lap2 = [gaussian_pyr[LEVEL - 1]]
    # for i in xrange(LEVEL - 1, 0, -1):
    #     lap2.append(gaussian_pyr[i-1] - expand(gaussian_pyr[i]))
    # print len(lap2)
    print len(lap)
    return lap
    # return lap2[::-1]

    # lpA = [gaussian_pyr[5]]
    # print "ading ", gaussian_pyr[5].shape
    # for i in xrange(LEVEL- 1, 0, -1):
    #     print i, gaussian_pyr[i].shape, gaussian_pyr[i - 1].shape
    #     GE = cv2.pyrUp(gaussian_pyr[i])
    #     L = cv2.subtract(gaussian_pyr[i - 1], GE)
    #     lpA.append(L)
    # print[ i.shape for i in lpA]
    #
    # return lpA[::-1]



def blendGray(img1, img2, mask):
    mask = mask/255  # binary
    p1 = mask * img1
    p2 = (1 - mask) * img2
    p = p1+p2
    p = np.clip(p, 0, 255)

    return p


def reconstruct(lap_pyr):
    laplacian_pyr = lap_pyr[:]
    c=0
    # for ii in range(len(laplacian_pyr)-1, 0, -1):
    for ii in range(len(laplacian_pyr)-2,-1,-1):
        smaller = laplacian_pyr[ii+1]
        bigger = laplacian_pyr[ii]
        new = (bigger + expand(smaller))
        new = np.clip(new, 0, 255)
        c+=1
        laplacian_pyr[ii] = new
    print "recons",c
    return new

    # lap_pyr = lap_pyr[::-1]
    # ls_ = lap_pyr[0]
    # for i in xrange(1,6):
    #     ls_ = expand(ls_)
    #     ls_ = cv2.add(ls_, lap_pyr[i])
    # return ls_[::-1]
    #
    # # return ls_



#
# ######### START OF OPENCV

# # generate Gaussian pyramid for A
# G = im1.copy()
# gpA = [G]
# for i in xrange(LEVEL):
#     G = cv2.pyrDown(G)
#     gpA.append(G)
#
# # generate Gaussian pyramid for B
# G = im2.copy()
# gpB = [G]
# for i in xrange(LEVEL):
#     G = cv2.pyrDown(G)
#     gpB.append(G)
#
# # generate Gaussian pyramid for mask
# G = msk.copy()
# gpM = [G]
# for i in xrange(LEVEL):
#     G = cv2.pyrDown(G)
#     gpM.append(G)
#
#
# # generate Laplacian Pyramid for A
# lpA = [gpA[LEVEL-1]]
# for i in xrange(LEVEL-1, 0, -1):
#     GE = cv2.pyrUp(gpA[i])
#     L = cv2.subtract(gpA[i - 1], GE)
#     lpA.append(L)
# print len(gpM)
# print len(lpA)
# print "!!"
# # generate Laplacian Pyramid for B
# lpB = [gpB[LEVEL-1]]
# for i in xrange(LEVEL-1, 0, -1):
#     GE = cv2.pyrUp(gpB[i])
#     L = cv2.subtract(gpB[i - 1], GE)
#     lpB.append(L)
#
# # [cv2.imshow("lev %s" %i, gpA[i]) for i in range(len(gpA))]
# # Now add left and right halves of images in each level
# LS = []
# gpM = gpM[::-1][1:]
# for la, lb, gm in zip(lpA, lpB, gpM):
#     gm = gm/255
#     ls = gm * la + (1-gm) * lb
#     LS.append(ls)
#
# # now reconstruct
# ls_ = LS[0]
# c=0
# for i in xrange(1, LEVEL):
#     c+=1
#     ls_ = cv2.pyrUp(ls_)
#     ls_ = cv2.add(ls_, LS[i])
# print "recons", c
#
# cv2.imshow('openCV', ls_)

# ######### END OP OPENCV

gaussian1 = genGaussianPyramid(im1, LEVEL)
gaussian2 = genGaussianPyramid(im2, LEVEL)
gaussian_msk = genGaussianPyramid(msk, LEVEL)


laplacian1 = genLaplacianPyramid(gaussian1, LEVEL)

laplacian2 = genLaplacianPyramid(gaussian2, LEVEL)
laplacian_m = genLaplacianPyramid(gaussian_msk, LEVEL)

print "my gaussian", len(gaussian2), [a.shape for a in gaussian2]
print "my laplacian", len(laplacian2), [a.shape for a in laplacian2]

# blend = [blendGray(laplacian1[i], laplacian2[i], gaussian_msk[i]) for i in range(LEVEL)]
blend=[]
for i in range(LEVEL):
    print "i",i
    blend.append(blendGray(laplacian1[i], laplacian2[i], gaussian_msk[i]))
print "!!"
# for i,b in enumerate(laplacian1):
#     cv2.imshow('pyr lev %s' % i, b)#ls_)

cv2.imshow("no pyr", blendGray(im1, im2, msk))
out = reconstruct(blend)
# exit(0)
# out= reconstruct(laplacian_m)
# out = blendGray(im1, im2, msk)

print np.amax(out), np.amin(out)
# img_s = expand(img, 2)
# img_b = reduce(img, 2)
# img_b = reduce(img, 2)
cv2.imshow('expanded my ', out)#ls_)
# cv2.imshow('expanded ', ls_)
# cv2.imshow('orin2', im2)
# cv2.imshow('orin1', im1)

cv2.waitKey(0)