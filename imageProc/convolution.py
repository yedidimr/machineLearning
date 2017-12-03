import numpy as np
import cv2  # open cv
import matplotlib.pylab as plt

img_path = "/home/student-5/Downloads/large-im2.jpg"
# img_path = "/home/student-5/Downloads/large-im1.jpg"


hpf_kernel_soft = np.array([[-1, -1, -1],
                            [-1,  8, -1],
                            [-1, -1, -1]])

hpf_kernel_hard = np.array([[-1, -1, -1, -1, -1],
                            [-1,  1,  2,  1, -1],
                            [-1,  2,  4,  2, -1],
                            [-1,  1,  2,  1, -1],
                            [-1, -1, -1, -1, -1]])

avg_kernel = np.ones((5,5),np.float32)/25


plot_i = 0
def plot(data, title):
    global plot_i
    plot_i += 1
    plt.subplot(3, 2, plot_i)
    plt.imshow(data, cmap='gray')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])



img = cv2.imread(img_path)

img_gaussian = cv2.GaussianBlur(img,(5,5),0)
img_edges = cv2.Canny(img,100,200)
img_hpf_soft = cv2.filter2D(img, -1, hpf_kernel_soft)  # using scipy: ndimage.convolve(img, hpf_kernel_soft)
img_hpf_hard = cv2.filter2D(img, -1, hpf_kernel_hard)
img_avg = cv2.filter2D(img, -1, avg_kernel)



plot(img, 'orig image')
plot(img_gaussian, 'Gaussian Filter')
print img.shape, img_gaussian.shape
plot(img_hpf_soft, 'HPF soft')
plot(img_hpf_hard, 'HPF hard')
plot(img_avg, 'Average Filter')
plot(img_edges, 'Canny Edge Filter')

plt.show()
