#define a CNN model for MNIST, same model as in the tensorflow tutorial

import tensorflow as tf
def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def place_holders():
	x = tf.placeholder("float", shape=[None, 784])
	y_ = tf.placeholder("float", shape=[None, 10])
	return x, y_

#x is an image tensor
def model(x):

	W_conv1 = weight_variable([5, 5, 1, 32], "W_conv1")
	b_conv1 = bias_variable([32], "b_conv1")

	x_image = tf.reshape(x, [-1,28,28,1])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
	b_conv2 = bias_variable([64], "b_conv2")

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([7 * 7 * 64, 1024], "W_fc1")
	b_fc1 = bias_variable([1024], "b_fc1")

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder("float")
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, 10], "W_fc2")
	b_fc2 = bias_variable([10], "b_fc2")

	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	variable_dict = {"W_conv1":W_conv1, "b_conv1":b_conv1, "W_conv2": W_conv2, 
		"b_conv2": b_conv2, "W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2":W_fc2, "b_fc2":b_fc2}
	return y_conv, keep_prob, variable_dict
