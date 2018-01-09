import numpy as np
import matplotlib.pylab as plt
import cv2

image_path1 = "/home/reut/Downloads/apple.jpg"
image_path2 = "/home/reut/Downloads/raccoon.png"

plot_i = 0
def plot(img, title, n_rows=4, n_cols=2):
    global plot_i
    assert plot_i < n_rows * n_cols
    plot_i += 1
    plt.subplot(n_rows, n_cols, plot_i)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])


def extract(image):
   """
   returns the magnitude and phase according to FFT
   """
   # calculate fft
   spectrum = np.fft.fft2(image)
   fshift = np.fft.fftshift(spectrum)  # to make the magnitude graph with the lower frequency in the middle

   # calculate phase and magnitude
   magnitude = np.abs(fshift)
   phase = np.angle(fshift)

   return magnitude, phase


def constract(phase, magnitude):
   """
   constrat an image according to it's phase and magnitude using inverse FFT
   """
   new_spectrum = magnitude * np.exp(1j * phase)

   # reverse the shift and FFT
   f_ishift = np.fft.ifftshift(new_spectrum)
   img_back = np.fft.ifft2(f_ishift)
   
   return np.abs(img_back)



img = cv2.imread(image_path1, 0)
img2 = cv2.imread(image_path2, 0)


# extract magnitude & phase from each image
magnitude, phase= extract(img)
magnitude2, phase2 = extract(img2)

# create a new image out of different magnitude and phase:
img_back = constract(phase, magnitude2)
img_back2 = constract(phase2, magnitude)


# prepare magniture to display - take the log to make magnitude visable on heatmap
visible_magnitude = 20*np.log(magnitude) 
visible_magnitude2 = 20*np.log(magnitude2) 


# display output
plot_i = 0
plot(img, 'Input img 1')
plot(img2, 'Input img 2')
plot(visible_magnitude, 'Magnitude Spectrum img 1')
plot(visible_magnitude2, 'Magnitude Spectrum img 2')

plot(phase, 'Phase Spectrum img 1')
plot(phase2, 'Phase Spectrum img 2')
plot(img_back, 'New img - phase of img1')
plot(img_back2, 'New img - phase of img2')
plt.show()
