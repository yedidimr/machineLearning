import numpy as np
import matplotlib.pylab as plt
from scipy import misc


image_path1 = "/home/student-5/Downloads/chess.png"
image_path2 = "/home/student-5/Downloads/apple2.png.png"

# image_path1 = "/home/student-5/Downloads/face.png"
# image_path2 = "/home/student-5/Downloads/aple.jpeg"

# read images
img = misc.imread(image_path1, flatten=True)
img2 = misc.imread(image_path2, flatten=True)

#resize by the smaller image
if (img2.size < img.size):
    img = misc.imresize(img, img2.shape )
if (img.size < img2.size):
    img2 = misc.imresize(img2, img.shape )

# calculate fft
spectrum = np.fft.fft2(img)
fshift = np.fft.fftshift(spectrum)  # to make the magnitude graph with the lower frequency in the middle

spectrum2 = np.fft.fft2(img2)
fshift2 = np.fft.fftshift(spectrum2)

# calculate phase and magnitude
magnitude = np.abs(fshift)
phase = np.angle(fshift)

magnitude2 = np.abs(fshift2)
phase2 = np.angle(fshift2)


# create a new image out of different magnitude and phase:

# calculate spectrum from magnitude and phase
new_spectrum = magnitude2 * np.exp(1j * phase)
new_spectrum2 = magnitude * np.exp(1j * phase2)

# reverse the shift and FFT
f_ishift = np.fft.ifftshift(new_spectrum)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

f_ishift2 = np.fft.ifftshift(new_spectrum2)
img_back2 = np.fft.ifft2(f_ishift2)
img_back2 = np.abs(img_back2)

# plot

plt.subplot(421)
plt.imshow(img, cmap='gray')
plt.title('Input Image1')
plt.xticks([])
plt.yticks([])

plt.subplot(422)
plt.imshow(img2, cmap='gray')
plt.title('Input Image2')
plt.xticks([])
plt.yticks([])


# show magnitude
plt.subplot(423)
magnitude_spectrum = 20*np.log(np.abs(fshift)) # log must be done to show correctly
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum of img1')
plt.xticks([])
plt.yticks([])

plt.subplot(424)
magnitude_spectrum2 = 20*np.log(np.abs(fshift2)) # log must be done to show correctly
plt.imshow(magnitude_spectrum2, cmap='gray')
plt.title('Magnitude Spectrum of img2')
plt.xticks([])
plt.yticks([])


# show phase
plt.subplot(425)
plt.imshow(phase, cmap='gray')
plt.title('Phase Spectrum img 1')
plt.xticks([])
plt.yticks([])

plt.subplot(426)
plt.imshow(phase2, cmap='gray')
plt.title('Phase Spectrum img 2')
plt.xticks([])
plt.yticks([])


plt.subplot(427)
plt.imshow(img_back, cmap='gray')
plt.title('New img - phase of img1')
plt.xticks([])
plt.yticks([])

plt.subplot(428)
plt.imshow(img_back2, cmap='gray')
plt.title('New img - phase of img2')
plt.xticks([])
plt.yticks([])

plt.show()









