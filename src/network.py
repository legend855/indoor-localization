
import tensorflow as tf
from data_formats import *
from __future__ import print_function



# tf.logging.set_verbosity(tf.logging.INFO)

# weights = tf.Variable(tf.random_normal([17786, 10], stddev=0.5), name="weights")
# biases = tf.Variable(tf.zeros([10]), name="biases")

# Parameters/ optimization variables 
learning_rate = 0.1
epochs = 10
batch_size = 128
display_step = 100


# Network parameters 
# input X: from measure two, each vector is approximately 17786
x = tf.placeholder(tf.float32, [None, 17786])

# Output, the (x, y) co-ordinates of the current predicted position
y = tf.placeholder(tf.float32, [None, 3])

# weights and biases

# from inputs to hidden layer
W1 = tf.Variable(tf.random_normal([17786, 300], stddev=0.5), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')

# from hidden layer to output 
W2 = tf.Variable(tf.random_normal([300, 3], stddev=0.5), name='W2')
b2 = tf.Variable(tf.ranodm_normal([3]), name='b2')

# Hidden layer out put 

hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.sigmoid(hidden_out)

# Outer layer 
y = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1-y) * tf.log(1 - y_clipped), axis=1))

optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

init_var = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Place this when launching the model 
#
#with tf.Session() as sess:
#	sess.run(init_var)
#	total_batch = int(len(
	# continue here with the model
	
	
	# how to define  neural network with 
