import matplotlib.pylab as plt
import cv2

img_path = "/home/reut/Downloads/puppy.jpg"


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


# import numpy as np
# from scipy import misc, signal
# import cv2
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


def quant(img, new_gray_level):
    """
    decrease image gray level from 8 to new_gray_level
    """
    scale = 256/(2**new_gray_level)
    img = img/scale      # reduce colors to the requested gray level
    return img * scale   # change back to uint8 - otherwise opencv won't show the image right




# load an image as grayscale
img = cv2.imread(img_path, 0)

# we can see that the numbers are uint8 - i.e 8 bits numbers (from 0 to 255)
print "numbers type", type(img[0,0])

# quantization
img2 = quant(img,2) # two levels of gray
img4 = quant(img,4) # four levels of gray

# show the images

plot = ImagePlot(n_rows=1, n_cols=3)
plot.add_img(img, 'original - 8 levels')
plot.add_img(img4, '4 levels')
plot.add_img(img2, '2 levels')
plot.show()
