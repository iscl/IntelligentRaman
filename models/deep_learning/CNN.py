'''
Created on Thu Nov 15

@author: Andy
'''

import tensorflow as tf
import load_data as ld
import numpy as np
import matplotlib.pyplot as plt

def weights_variables(shape):
    filter_weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return filter_weight

def bias_variables(shape):
    biases = tf.Variable(tf.constant(0.1, shape=shape))
    return biases

def conv1d(x, W, stride, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=stride, padding=padding)

def max_pool(x, ksize, stride, padding='SAME'):
    return tf.nn.max_pool(x, ksize=ksize, strides=stride, padding=padding)

xs = tf.placeholder(tf.float32, [None, 1866], name='intensity')
ys = tf.placeholder(tf.float32, [None, 4], name='molecule')

x_image = tf.reshape(xs, [-1, 1, 1866, 1])

# block1
W_conv1 = weights_variables([1, 125, 1, 32])
b_conv1 = bias_variables([32])
conv1 = conv1d(x_image, W_conv1, [1, 1, 1, 1], padding='VALID')
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b_conv1))
pool1 = max_pool(relu1, [1, 1, 2, 1], [1, 1, 2, 1])

# block2
W_conv2 = weights_variables([1, 125, 32, 64])
b_conv2 = bias_variables([64])
conv2 = conv1d(pool1, W_conv2, [1, 1, 1, 1], padding='VALID')
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b_conv2))
pool2 = max_pool(relu2, [1, 1, 2, 1], [1, 1, 2, 1])

# block3
W_conv3 = weights_variables([1, 125, 64, 128])
b_conv3 = bias_variables([128])
conv3 = conv1d(pool2, W_conv3, [1, 1, 1, 1], padding='VALID')
relu3 = tf.nn.relu(tf.nn.bias_add(conv3, b_conv3))
pool3 = max_pool(relu3, [1, 1, 2, 1], [1, 1, 2, 1])

# block4
W_conv4 = weights_variables([1, 125, 128, 256])
b_conv4 = bias_variables([256])
conv4 = conv1d(pool3, W_conv4, [1, 1, 1, 1], padding='VALID')
relu4 = tf.nn.relu(tf.nn.bias_add(conv4, b_conv4))
pool4 = max_pool(relu4, [1, 1, 2, 1], [1, 1, 2, 1])

# fc1
pool_shape = pool4.get_shape().as_list()
nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
reshaped = tf.reshape(pool4, [-1, nodes])
# reshaped = tf.reshape(pool4, [-1, 1 * 351 * 256])
W_fc1 = weights_variables([nodes, 2048])
b_fc1 = bias_variables([2048])
fc1 = tf.nn.relu(tf.matmul(reshaped, W_fc1) + b_fc1)

# fc2
W_fc2 = weights_variables([2048, 4])
b_fc2 = bias_variables([4])
fc2 = tf.matmul(fc1, W_fc2) + b_fc2

correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# prediction
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc2, labels=tf.argmax(ys, 1))

cross_entropy_mean = tf.reduce_mean(cross_entropy)

train_step = tf.train.AdamOptimizer().minimize(cross_entropy_mean)

xs_data, ys_label = ld.load_data('../../datasets/train.csv.bz2')
xs_test_data, ys_test_label = ld.load_data('../../datasets/test.csv.bz2')

batch_size = 100
num_step = 10001

step_for_curve = np.zeros(num_step)
loss_for_vurve = np.zeros_like(step_for_curve)

# rand_index = np.random.choice(699, size=batch_size)
#
# # offset = (step * batch_size) % (ys_label.shape[0] - batch_size)
# batch_xs = xs_data[rand_index]
# batch_ys = ys_label[rand_index]
#
# print("batch_xs:", batch_xs)
# print("batch_ys:", batch_ys)

# offset = (1 * batch_size) % (ys_label.shape[0] - batch_size)
# print('offset:', offset)
# batch_xs = xs_data[offset:(offset + batch_size), :]
# batch_ys = ys_label[offset:(offset + batch_size), :]
# fetch_dict = {xs: batch_xs, ys: batch_ys}
# print('feed_dict:', fetch_dict)

# print('slected xs:', xs[0: 10, :])
# print('sleceted ys:', ys[0: 10, :])
#
# batch_xs = xs[0: 10, :]
#
# fetch_dict = {xs: batch_xs}
#
# print('feed_dict:', fetch_dict)


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    test_feed = {xs: xs_test_data, ys: ys_test_label}

    for step in range(num_step):
        if(step % 1000) == 0:
            validate_acc = sess.run(accuracy, feed_dict=test_feed)
            print("After %d training step(s), validation accuracy is %g" % (step, validate_acc))

        rand_index = np.random.choice(699, size=batch_size)

        batch_xs = xs_data[rand_index]
        batch_ys = ys_label[rand_index]

        # offset = (step * batch_size) % (ys_label.shape[0] - batch_size)
        # batch_xs = xs_data[offset:(offset + batch_size), :]
        # batch_ys = ys_label[offset:(offset + batch_size), :]

        fetch_dict = {xs: batch_xs, ys: batch_ys}

        _, loss = sess.run([train_step, cross_entropy_mean], feed_dict=fetch_dict)
        step_for_curve[step] = step
        loss_for_vurve[step] = loss

        print('The', step, 'step finshed. The loss is', loss)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(step_for_curve, loss_for_vurve, label='loss')
ax.set_xlabel('step')
ax.set_ylabel('loss')
fig.suptitle('Cross_Entorpy')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels=labels)
plt.show()
