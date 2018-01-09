import numpy as np
import cv2  # open cv
import matplotlib.pylab as plt

class ImagePlot(object):
    def __init__(self, n_rows, n_cols):
        self._plot_i = 0
        self.n_rows = n_rows
        self.n_cols = n_cols

    def add_img(self, img, title):
        assert self._plot_i < self.n_rows * self.n_cols
        self._plot_i += 1
        plt.subplot(self.n_rows, self.n_cols, self._plot_i)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

    def show(self):
        plt.show()
        self._plot_i = 0




# the function cv2.filter2D performs convolution between the given image and kernel.

# Define the filters (kernels):

# blur filter - 
#      output image will be a blurred version of the original.
# kernel replaces each pixel by the avg of its 3X3 patch. Sometimes gaussian kernel is used.
avg = 1/9. * np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])  # equal to: avg = np.ones((3, 3), np.float32) / 9

# vertical edge detection kernel - 
#      output image will show the vertical edges in the image (and not the image itself)
# kernel - shows the difference between the right and the left pixels. 
# intuation - egde appears where we have color changes. Therefore, if the colors to the right and to the left
#             of a pixel are the same, the difference is 0 (or small number) and the pixel is black.
#             bigger difference --> whiter pixel, and the egde is sharper.
ver_dev = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])


# horizontal edge detection kernel
# similar to vertical edge - but here we check the difference between the pixels above and below.
hor_dev = np.array([[-1, -1, -1],
                    [0,  0,  0],
                    [1,  1,  1]])

# full detection kernel
# a combination of the horizontal & vertical edge detection.
# intuation - when the current pixel and it's (8) surrunding pixels are the same color - shows black.
#             shows white only when there is a color change.
corner = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])


# sharpen filter -
#       output image will be a sharpen version of the original.
# kernel makes the current pixek 5 times bigger and reducces the pixels above, below, left, right.
#       therefore the pixel keeps it's value only when it is the same as the pixels above, below, left, right .
#       otherwise the "contrast value" is added to the pixel.
sharp = np.array([[0, -1, 0],
                 [-1,  5, -1],
                 [0,  -1,  0]])
    
# general intuation - when the sum of the kernel is 0, the kernel usually shows difference (derivation) such as edge
# detection, corner detection, e.t.c.


# load the image
img_path = "/home/reut/Downloads/butterfly.jpg"
img = cv2.imread(img_path,0)

# apply convolution
ver_img =    cv2.filter2D(img, -1, ver_dev)
hor_img =    cv2.filter2D(img, -1, hor_dev)
sharpen =    cv2.filter2D(img, -1, sharp)
blur_img =   cv2.filter2D(img, -1, avg)
corner_img = cv2.filter2D(img, -1, corner)

# show the output
plot = ImagePlot(n_rows = 3, n_cols = 2)
plot.add_img(img, 'Orignal Image')
plot.add_img(ver_img, 'Horizontal Edge Detection')
plot.add_img(sharpen, 'Sharpen')
plot.add_img(hor_img, 'vertical Edge detection')
plot.add_img(blur_img, 'Blurred')
plot.add_img(corner_img, 'Edge Detection')
plot.show()



# # openCV built in convulotions filters
# img_gaussian = cv2.GaussianBlur(img,(5,5),0)
# img_edges = cv2.Canny(img,100,200)