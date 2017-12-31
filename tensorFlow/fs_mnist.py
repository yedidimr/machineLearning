import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

EPOCHS = 10
BATCH_SIZE = 128


def LeNet(x, output_size = 10):
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1': 6,
        'layer_2': 16,
        'layer_3': 120,
        'layer_f1': 84
    }

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    #Activation.
    conv1 = tf.nn.relu(conv1)

    #Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape=(256, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_w) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = output_size.
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84, output_size), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(output_size))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


mnist = mnist_data.read_data_sets("data", reshape=False) #, one_hot=True, validation_size=0)
#exract train 0 or 1
# zeros_indices = mnist.train.labels[:,0] == 1  #for one hot
# ones_indices = mnist.train.labels[:,1] == 1
zeros_indices = mnist.train.labels == 0
ones_indices = mnist.train.labels == 1
zeros_ones_indices = zeros_indices | ones_indices
train_images = mnist.train.images[zeros_ones_indices ]
train_labels = mnist.train.labels[zeros_ones_indices ]
X_train, y_train = shuffle(train_images, train_labels)

#exract test 0 or 1
# zeros_indices = mnist.test.labels[:,0] == 1
# ones_indices = mnist.test.labels[:,1] == 1
zeros_indices = mnist.test.labels == 0
ones_indices = mnist.test.labels == 1
zeros_ones_indices = zeros_indices | ones_indices
X_test = mnist.test.images[zeros_ones_indices ]
y_test = mnist.test.labels[zeros_ones_indices ]

#extract validatoion 0 or 1
# zeros_indices = mnist.validation.labels[:,0] == 1
# ones_indices = mnist.validation.labels[:,1] == 1
zeros_indices = mnist.validation.labels == 0
ones_indices = mnist.validation.labels == 1
zeros_ones_indices = zeros_indices | ones_indices
X_validation = mnist.validation.images[zeros_ones_indices ]
y_validation = mnist.validation.labels[zeros_ones_indices ]


# X_train, y_train           = mnist.train.images, mnist.train.labels
# X_validation, y_validation = mnist.validation.images, mnist.validation.labels
# X_test, y_test             = mnist.test.images, mnist.test.labels



output_size = 2 # 0 or 1

mnist = mnist_data.read_data_sets("data", reshape=False)#, one_hot=True , validation_size=0)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.int32, [None])
# correct answers will go herey = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, output_size)

rate = 0.001

logits = LeNet(x, output_size)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)

# optimizer = tf.train.AdamOptimizer(learning_rate = rate)
# training_operation = optimizer.minimize(loss_operation)

starter_learning_rate = rate
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 800, 0.96, staircase=True)
training_operation = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_operation, global_step=global_step)

#model evaluation

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")

#evaluate
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))