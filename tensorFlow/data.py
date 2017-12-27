import time
import tensorflow as tf
import numpy as np
import pickle
import os
from PIL import Image
import os



###################################
############  tiny image net  #####
###################################
TINY_IMAGE_NET_CLASS_NUM = 10 # 200
TINY_IMG_HOME_DIR = '/home/student-5/Downloads/tiny-imagenet-200'
tiny_img_net_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

#taken from https://github.com/miquelmarti/tiny-imagenet-classifier/blob/master/val_load.py
def get_annotations_map():
    valAnnotationsPath = os.path.join(TINY_IMG_HOME_DIR, 'val/val_annotations.txt')
    valAnnotationsFile = open(valAnnotationsPath, 'r')
    valAnnotationsContents = valAnnotationsFile.read()
    valAnnotations = {}

    for line in valAnnotationsContents.splitlines():
        pieces = line.strip().split()
        valAnnotations[pieces[0]] = pieces[1]

    return valAnnotations

# taken from : https://github.com/miquelmarti/tiny-imagenet-classifier/blob/master/load_images.py
def load_images(path, num_classes):
    # Load images

    print('Loading ' + str(num_classes) + ' classes')

    X_train = np.zeros([num_classes * 500, 3, 64, 64], dtype='uint8')
    y_train = np.zeros([num_classes * 500], dtype='uint8')

    trainPath = path + '/train'

    print('loading training images...');

    i = 0
    j = 0
    annotations = {}
    for sChild in os.listdir(trainPath):
        sChildPath = os.path.join(os.path.join(trainPath, sChild), 'images')
        annotations[sChild] = j
        for c in os.listdir(sChildPath):
            X = np.array(Image.open(os.path.join(sChildPath, c)))
            if len(np.shape(X)) == 2:
                X_train[i] = np.array([X, X, X])
            else:
                X_train[i] = np.transpose(X, (2, 0, 1))
            y_train[i] = j
            i += 1
        j += 1
        if (j >= num_classes):
            break

    print('finished loading training images')

    val_annotations_map = get_annotations_map()

    X_test = np.zeros([num_classes * 50, 3, 64, 64], dtype='uint8')
    y_test = np.zeros([num_classes * 50], dtype='uint8')

    print('loading test images...')

    i = 0
    testPath = path + '/val/images'
    for sChild in os.listdir(testPath):
        if val_annotations_map[sChild] in annotations.keys():
            sChildPath = os.path.join(testPath, sChild)
            X = np.array(Image.open(sChildPath))
            if len(np.shape(X)) == 2:
                X_test[i] = np.array([X, X, X])
            else:
                X_test[i] = np.transpose(X, (2, 0, 1))
            y_test[i] = annotations[val_annotations_map[sChild]]
            i += 1
        else:
            pass

    print('finished loading test images') + str(i)

    return X_train, y_train, X_test, y_test


#taken from ira & shachar
def prepare_imagenet_data():
    from sklearn import preprocessing
    path = TINY_IMG_HOME_DIR
    X_train, y_train, X_test, y_test = load_images(path, TINY_IMAGE_NET_CLASS_NUM)
    lb = preprocessing.LabelBinarizer()
    y_train_onehot = lb.fit_transform(y_train)
    y_test_onehot = lb.fit_transform(y_test)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    return X_train, y_train_onehot, X_test, y_test_onehot, 64, TINY_IMAGE_NET_CLASS_NUM


###################################
############  cifar  #####
###################################



########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
CIFAR_DATA_PATH = "/home/student-5/Downloads"

# URL for the data-set on the internet.
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of files for the training-set.
_num_files_train = 5

# Number of images for each batch-file in the training-set.
_images_per_file = 10000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file

########################################################################
# Private functions for downloading, unpacking and loading data-files.


def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.
    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]


def _get_file_path(filename=""):
    """
    Return the full path of a data-file for the data-set.
    If filename=="" then return the directory of the files.
    """

    return os.path.join(CIFAR_DATA_PATH, "cifar-10-batches-py/", filename)


def _unpickle(filename):
    """
    Unpickle the given file and return the data.
    Note that the appropriate dir-name is prepended the filename.
    """

    # Create full path for the file.
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file)# encoding='bytes')

    return data


def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])

    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.



def load_class_names():
    """
    Load the names for the classes in the CIFAR-10 data-set.
    Returns a list with the names. Example: names[3] is the name
    associated with class-number 3.
    """

    # Load the class-names from the pickled file.
    raw = _unpickle(filename="batches.meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names


def load_training_data():
    """
    Load all the training-data for the CIFAR-10 data-set.
    The data-set is split into 5 data-files which are merged here.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_test_data():
    """
    Load all the test-data for the CIFAR-10 data-set.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    images, cls = _load_data(filename="test_batch")

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)



def prepare_cifar_data():
    train_images, _, train_cls_vec = load_training_data() #cifar
    test_images, _, test_cls_vec = load_test_data() #cifar

    return  train_images, train_cls_vec, test_images, test_cls_vec, 32, 10