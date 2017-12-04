import cv2
import numpy as np
import matplotlib.pylab as plt
from scipy import misc

# read image
img_path = "/home/student-5/Downloads/im1.jpg"
im = cv2.imread(img_path, 0)
im = im.astype(np.float64)/255

cv2.imshow("orig img", im)

# create noise matrix
noise = np.random.rand(im.shape[0], im.shape[1])
cv2.imshow("noise", noise)

# multiply to create a noise image
noise_img = np.dot(noise, im)
cv2.imshow("noisy img", noise_img/np.amax(noise_img))  # normalize



# find noise inverse using SVD
U, s, Vt = np.linalg.svd(noise)


S = np.linalg.inv(np.diag(s))
Ut = np.transpose(U)
V = np.transpose(Vt)

noise_inverse = np.dot(np.dot(V, S), Ut)

reverted_im = np.dot(noise_inverse, noise_img)
#
# n = s.shape[0]
# s_sorted = np.argsort(s)
# rms=[]
# for k in range(n):
#     s[s_sorted[:n-k]] = 0  # take indices of n-k smallest
#     rms.append(np.sqrt(np.mean(np.square(im - reverted_im))))
# plt.plot(range(k), rms)
# plt.show()
# revert original image using inverted noise matrix
cv2.imshow("reverted img using SVD", reverted_im )

cv2.waitKey(0)







