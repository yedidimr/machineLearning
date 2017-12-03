import cv2
import numpy as np
import matplotlib.pylab as plt
from scipy import misc



plot_i = 0
# plt.ion() # turn on interactive mode
def plot(data, title):
    global plot_i
    plot_i += 1
    # plt.subplot(2, 3, plot_i)
    dpi = 80
    height, width = data.shape

    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)


    # plt.imshow(data, cmap='gray')
    ax = fig.add_axes([0, 0, 1, 1])
    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(data, cmap='gray')
    # fig.suptitle(title)


class Resize(object):
    def __init__(self, src, fx=0, fy=0, dsize=0):
        """

        :param src: input image
        :param dsize: output image size; Either dsize or both fx and fy must be non-zero.
        :param fx: scale factor along the horizontal axis
        :param fy: scale factor along the vertical axis
        """
        assert ((fx * fy != 0) or  dsize) # Either dsize or both fx and fy must be non-zero.
        self.src = src
        self.src_shape = src.shape
        if dsize:
            self.fx = dsize[0]/self.src_shape[0]
            self.fy = dsize[1]/self.src_shape[1]
        else:
            self.fy = fy
            self.fx = fx

        self.dst_shape = (int(self.src_shape[0] * self.fx), int(self.src_shape[1] * self.fy))
        self.dst = np.zeros(self.dst_shape)

class NearestNeighbour(Resize):
    """
    from wiki:
    One of the simpler ways of increasing image size is nearest-neighbor interpolation,
    replacing every pixel with multiple pixels of the same color:
    The resulting image is larger than the original, and preserves all the original detail,
    but has (generally undesirable) jaggedness. Diagonal lines, for example, show a "stairway" shape.
    on shrinking  pixels are thrown
    :param img: the image matrix
           mult: the resizing factor
    :return:
            image  which is  mult times larger
    """

    def resize(self):
        w,h = self.src_shape  # original width and height
        for i in range(self.dst_shape[0]):
            for j in range(self.dst_shape[1]):
                self.dst[i,j] = self.src[int(i/self.fx), int(j/self.fy)]

        return  self.dst/255.




class Bilinear(Resize):
    """
        Bilinear image scaling
        grayscale
        """

    def resize(self):
        w,h = self.src_shape  # original width and height
        for i in range(self.dst_shape[0]):
            for j in range(self.dst_shape[1]):
                x = int(self.fx * i)
                y = int(self.fy * j)
                x_diff = (self.fx * i) - x
                y_diff = (self.fy * j) - y
                index = y * w + x -10

                # range is 0 to 255 thus bitwise AND with 0xff
                print self.src[index], type(self.src[index])
                A = self.src[index] & 0xff
                B = self.src[index+1] & 0xff
                C = self.src[index+w] & 0xff
                D = self.src[index+w+1] & 0xff

                # Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + Dwh
                gray = int(
                    A * (1-x_diff) * (1-y_diff) +  B * (x_diff) * (1-y_diff) +
                    C * (y_diff) * (1-x_diff)   +  D * (x_diff * y_diff)
                )

                self.dst[i,j] = gray

        return self.dst;

image_path1 = "/home/student-5/Downloads/large-im2.jpg"
image_path1 = "/home/student-5/Downloads/aple.jpeg"
image_path1 = "/home/student-5/Downloads/large-im1.jpg"

# aspet ratio
w=2
h=2

# load image
img = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
# img = 1- np.diag(np.ones(400)) # create image of black diagonal


nn = NearestNeighbour(img, w, h)
nn_img = nn.resize()
cv2.imshow('my - Nearest Neighbour %s'% str(nn_img.shape), nn_img)

# bliner_resizing = Bilinear(img, w, h)
# bilinear_img = bliner_resizing.resize()
# cv2.imshow('my - Bilinear', bilinear_img)




new_shape = (int(img.shape[1]*w), int(img.shape[0]*h))
print "new img shape is", new_shape, cv2.INTER_LINEAR

cv2.imshow('original image %s'% str(img.shape), img)

cv_nn_img = cv2.resize(img, new_shape ,  interpolation=cv2.INTER_NEAREST)
cv2.imshow('openCV - Nearest Neighbour %s' % str(cv_nn_img.shape), cv_nn_img)

cv_linear_img = cv2.resize(img, new_shape ,  interpolation=cv2.INTER_LINEAR)
cv2.imshow('openCV - Bilinear %s' % str(cv_linear_img.shape), cv_linear_img)



# converting formats
img = cv2.imread(image_path1)#, cv2.IMREAD_GRAYSCALE)
cv2.imshow('org' , img)

img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('grayscale' , img_g)

img_h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv' , img_h)


cv2.waitKey(0)









