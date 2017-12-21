import tensorflow as tf
import numpy as np
import pickle
import os

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_path = "/home/student-5/Downloads"

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

    return os.path.join(data_path, "cifar-10-batches-py/", filename)


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




images, cls_res, labels = load_training_data()
print images.shape, "images.shape"
print labels.shape, "labels.shape"
# images shape is  (50000, 32, 32, 3)  (50000 images of 32X32, 3 channels RGB)
# cls_res shape is  (50000,)  contains the number of class for each sample (from 0 to 9)
# labels shape is  (50000,10) holdes a binary vector for each sample with 1 in the index representing the class number




x_ = tf.placeholder(tf.float32, shape=[None,32,32,3], name = 'x-input') # input: 32X32 image, RGB channels
y_ = tf.placeholder(tf.float32, shape=[None,10], name = 'y-input')  # output: 10 optional classes




W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 16]), name ="weight1")  # 5X5 filter, 16 filters, 3 input channgels (depth)
B1 = tf.Variable(tf.zeros([16]), name = "bias1")

W2 = tf.Variable(tf.truncated_normal([5, 5, 16, 20]), name ="weight2")  # 5X5 filter, 20 filters,
B2 = tf.Variable(tf.zeros([20]), name = "bias2")

W3 = tf.Variable(tf.truncated_normal([5, 5, 20, 20]), name ="weight3")  # 5X5 filter, 20 filters,
B3 = tf.Variable(tf.zeros([20]), name = "bias3")

W4 = tf.Variable(tf.truncated_normal([320, 10]), name ="weight4")  # 5X5 filter, 20 filters,
B4 = tf.Variable(tf.zeros([10]), name = "bias4")

stride = 1
with tf.name_scope("hidden1") as scope:
    out1 = tf.nn.relu(tf.nn.conv2d(x_, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)  #activation - relu, padding -
    pool1 = tf.nn.max_pool(out1, ksize = [1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope("hidden2") as scope:
    out2 = tf.nn.relu(tf.nn.conv2d(pool1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)  #activation - relu, padding -
    pool2 = tf.nn.max_pool(out2, ksize = [1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope("hidden3") as scope:
    out3 = tf.nn.relu(tf.nn.conv2d(pool2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)  #activation - relu, padding -
    pool3 = tf.nn.max_pool(out3, ksize = [1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope("hidden4") as scope:
    # flatten pool 3 to be ready for fully connected NN
    print "pool shape before", pool3.shape
    pool3 = tf.reshape(pool3, [-1, pool3.shape[1] * pool3.shape[2] * pool3.shape[3]])
    print "pool shape after", pool3.shape
    print "YAY"
    logits = tf.matmul(pool3, W4) + B4
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_) # cost function? activation function?



train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

Y = tf.nn.softmax(logits)
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(y_, 1))
print "is-correct", is_correct.shape
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph_def)

for i in range(7000):
    indices = np.random.randint(0, images.shape[0], 100)
    X_data, Y_data = images[indices, :, :, :], labels[indices]

    sess.run(train_step, feed_dict={x_: X_data, y_: Y_data})

    if i % 100 == 0:
        # ce = sess.run(cross_entropy, feed_dict={x_: X_data, y_: Y_data})
        # pred = sess.run(Y, feed_dict={x_: X_data, y_: Y_data})
        a = sess.run(accuracy, feed_dict={x_: X_data, y_: Y_data})
        print "accuracy", a
        # print "cross_entropy", ce
import pdb

pdb.set_trace()


