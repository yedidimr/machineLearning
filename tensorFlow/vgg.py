import os
import shutil
import time
from PIL import Image

import data
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

tensorFlow_file_path = "/home/student-5/PycharmProjects/machineLearning/tensorFlow/logs/vgg"

# try:
#     os.remove(tensorFlow_file_path)
# except OSError:
#     pass

os.system("rm -r %s/*" % tensorFlow_file_path)


def batch_generator(images, classes, batch_size):
    indices = range(len(images))
    while True:
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            yield images[batch_indices, :, :, :], classes[batch_indices]


def get_batch(x, y, batch_size):
    indices = np.random.randint(0, x.shape[0], batch_size)
    return x[indices, :, :, :], y[indices]


def conv(input, filters_count, filter_size=3, scope_name="conv", stride=1):
    with tf.name_scope(scope_name) as scope:  # todo maybe remov scope down
        # weights = tf.get_variable('weights', [filter_size, filter_size, from_d, to_d], initializer=tf.truncated_normal_initializer(stddev=0.01))
        # bias = tf.Variable(tf.zeros([to_d]), name="bias")   # todo if doesnt work try this: tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')  todo remove
        # out = tf.nn.relu(tf.nn.conv2d(input, weights, strides=[1, stride, stride, 1], padding='SAME') + bias)

        # can do three lines in one like this:
        # stddev = np.sqrt(2 / (filter_size * filter_size * int(input.shape[3])))
        out = tf.layers.conv2d(input, filters=filters_count,
                               kernel_size=(filter_size, filter_size),
                               padding='same',
                               activation=tf.nn.relu,
                               kernel_initializer=layers.xavier_initializer(),
                               # kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-04),
                               use_bias=True,
                               bias_initializer=tf.constant_initializer(0.0),
                               data_format='channels_first',
                               name=None)

    return out


def pool(input, scope_name="pool", ksize=2, strides=2):
    with tf.name_scope(scope_name) as scope:
        out = layers.max_pool2d(input, ksize, strides, padding='SAME', data_format='NCHW')
    return out


def fully_connected(input, output_size, activation=None, scope_name="fc"):
    with tf.name_scope(scope_name) as scope:
        # weights = tf.Variable(tf.truncated_normal([from_d, to_d], stddev=0.1), name="weights")
        # bias = tf.Variable(tf.zeros([to_d]), name="bias")        # todo if doesnt work try ones: tf.Variable(tf.constant(1.0,
        # logits = tf.matmul(input, weights) + bias
        # out = tf.nn.relu(logits)
        stddev = np.sqrt(2 / int(input.shape[1]))
        out = tf.layers.dense(input, output_size, activation=activation,
                              # kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                              # kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                              kernel_initializer=layers.xavier_initializer(),
                              # kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-04),
                              use_bias=True,
                              bias_initializer=tf.constant_initializer(0.0),
                              name=None)

    return out


print "loading data"
t0 = time.time()
train_images, train_cls_vec, test_images, test_cls_vec, img_size, labels_count = data.prepare_imagenet_data()
test_images = test_images
t1 = time.time()
print "data took %d" % (t1 - t0)

x_ = tf.placeholder(tf.float32, [None, 3, img_size, img_size], name='images')
# x_ = tf.image.resize_images(x_, [img_size, img_size])
y_ = tf.placeholder(tf.float32, shape=[None, labels_count], name='labels')

# the defult filter size is 3X3
out = conv(x_, 64, scope_name="conv1.1")
out = conv(out, 64, scope_name="conv1.2")
out = pool(out, scope_name="pool1")

out = conv(out, 128, scope_name="conv2.1")
out = conv(out, 128, scope_name="conv2.2")
out = pool(out, scope_name="pool2")

out = conv(out, 256, scope_name="conv3.1")
out = conv(out, 256, scope_name="conv3.2")
out = conv(out, 256, scope_name="conv3.3", filter_size=1)
out = pool(out, scope_name="pool3")

out = conv(out, 512, scope_name="conv4.1")
out = conv(out, 512, scope_name="conv4.2")
out = conv(out, 512, scope_name="conv4.3", filter_size=1)
out = pool(out, scope_name="pool4")

out = conv(out, 512, scope_name="conv5.1")
out = conv(out, 512, scope_name="conv5.2")
out = conv(out, 512, scope_name="conv5.3", filter_size=1)
out = pool(out, scope_name="pool5")

# flatten before FC layer
shape = out.shape[1] * out.shape[2] * out.shape[3]
print shape, type(shape)

# print pool5.shape
out = tf.reshape(out, [-1, shape])
# print pool5.shape
out = fully_connected(out, 1024, activation=tf.nn.relu, scope_name="fc1")

prob = tf.placeholder_with_default(1.0, shape=())
with tf.name_scope('dropout'):
    out = tf.nn.dropout(out, prob)

out = fully_connected(out, 1024, activation=tf.nn.relu, scope_name="fc2")

with tf.name_scope('dropout'):
    out = tf.nn.dropout(out, prob)

out = fully_connected(out, labels_count, scope_name="fc3")

##################### calc_accuracy
probs = tf.nn.softmax(out)
is_correct = tf.equal(tf.argmax(probs, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
tf.summary.scalar('accuracy', accuracy)

##################### loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_)  # cost function? activation function?
cross_entropy = tf.reduce_mean(cross_entropy)
tf.losses.add_loss(cross_entropy)
# loss = tf.losses.get_total_loss() #change 1. if disables = no regulation
loss = cross_entropy  #change 2. if enabled = no regulation
# reg_loss = tf.losses.get_regularization_loss()

tf.summary.scalar('cross_entropy_loss', loss)
# tf.summary.scalar('regulization_loss', reg_loss)

##################### learning step (GDC optimizer with exponential decay for learning rate)
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.0001  # change to 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 800, 0.96, staircase=True)
tf.summary.scalar('learning_rate', learning_rate)
# learning_rate = starter_learning_rate
# Passing global_step to minimize() will increment it at each step.
# learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
learning_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

##################### training

merged = tf.summary.merge_all()
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)

t0 = time.time()


accuracy_test = tf.placeholder(tf.float32, shape=())
cross_entropy_loss_test = tf.placeholder(tf.float32, shape=())
summary_ac = tf.summary.scalar('accuracy', accuracy_test)
summary_ce = tf.summary.scalar('cross_entropy_loss', cross_entropy_loss_test)
merge_test_summery = tf.summary.merge([summary_ac, summary_ce])

mean = np.array([116.779, 103.939, 123.68]).reshape([1, 3, 1, 1])
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter("logs/vgg/train", sess.graph)
    test_writer = tf.summary.FileWriter("logs/vgg/test")
    sess.run(tf.global_variables_initializer())
    i = 1
    for batch_x, batch_y in batch_generator(train_images, train_cls_vec, 256):
        if i % 100000 == 0:
            break
        i += 1
        batch_x = batch_x - mean

        sess.run(learning_step, feed_dict={x_: batch_x, y_: batch_y,
                                           prob: 0.75})  # fedforwarrd o f a single batch theouth the NN + backprop

        if i % 100 == 0:
            cost_empirical, accuracy_empirical, summary = sess.run([cross_entropy, accuracy, merged],
                                                                   feed_dict={x_: batch_x,
                                                                              y_: batch_y})  # calculte the cross entropy + accuracy of the last learning step
            # summary = sess.run(merged, feed_dict={x_: batch_x, y_: batch_y})
            train_writer.add_summary(summary, i)
            print(
                "Iteration {}, time passed: {}, train cost: {}, train accuracy: {}".format(i, time.time() - t0,
                                                                                           cost_empirical,
                                                                                           accuracy_empirical))
        # perform test batch

        if i % 800 == 0:
            # batch_x_test, batch_y_test = test_images, test_cls_vec
            batch_acc_list, batch_loss_list = [], []
            for j, (test_batch_x, test_batch_y) in enumerate(batch_generator(test_images, test_cls_vec, 256)):
                test_batch_x = test_batch_x - mean
                cost_empirical, accuracy_empirical = sess.run([cross_entropy, accuracy],
                                                              feed_dict={x_: test_batch_x,
                                                                         y_: test_batch_y})
                batch_acc_list.append(accuracy_empirical)
                batch_loss_list.append(cost_empirical)
                if j >= 39:
                    break

            test_acc = reduce(lambda x, y: x + y, batch_acc_list) / float(len(batch_acc_list))  # equal to 1.0*sum(batch_acc_list)/len(batch_acc_list)
            test_loss = reduce(lambda x, y: x + y, batch_loss_list) / float(len(batch_loss_list))

            merge = sess.run([merge_test_summery], feed_dict={
                accuracy_test: test_acc,
                cross_entropy_loss_test: test_loss
            })
            test_writer.add_summary(merge[0], i)
            print("Test cost: {}, Test accuracy: {}".format(test_loss, test_acc))

import pdb

pdb.set_trace()