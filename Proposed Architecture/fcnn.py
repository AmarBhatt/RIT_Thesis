import matplotlib.pyplot as plt
import tensorflow as tf
#import pandas as pd
import numpy as np
#from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
#import dataset
import random
import os.path
import h5py
from DataSetGenerator.maze_generator import *
	
# Convolutional Layer 1.
filter_size1 = 3 
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64
	
# Fully-connected layer.
fc_size = 128			 # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 1

# image dimensions (only squares for now)
img_size = 100

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info

classes = [0,1,2,3,4]
num_classes = len(classes)

# batch size
batch_size = 16

# validation split
validation_size = .2

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping





train_path='training_data'
test_path='testing_data'


#data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
#test_images, test_ids = dataset.read_test_set(test_path, img_size,classes)

f_data = "random"
f_file = "random.h5"

num = 80
num2 = 100
num_epochs = 10 #20
episodes = 50

if not os.path.isfile(f_file):

	episode_lengths = []
	episode_start = []
	episode_total = 0
	image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(0),img_size)

	x_train = image_set
	y_train = action_set

	episode_lengths.append(action_set.shape[0])
	episode_start.append(episode_total)
	episode_total += image_set.shape[0]
	for i in range(1,num):
		print(i)
		image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(i),img_size)
		x_train = np.append(x_train,image_set,axis=0)
		y_train = np.append(y_train,action_set, axis=0)
		episode_lengths.append(action_set.shape[0])
		episode_start.append(episode_total)
		episode_total += image_set.shape[0]
	#print(episode_lengths)
	print(x_train.shape)
	print(y_train.shape)

	image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(num),img_size)

	x_test = image_set
	y_test = action_set

	for i in range(num+1,num2):
		image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(i),img_size)
		x_test = np.append(x_test,image_set,axis=0)
		y_test = np.append(y_test,action_set, axis=0)

	print(x_test.shape)
	print(y_test.shape)
	
	with h5py.File(f_file,'w') as hf:
		hf.create_dataset("x_train",data=x_train, shape = x_train.shape)
		hf.create_dataset("y_train",data=y_train, shape = y_train.shape)
		hf.create_dataset("episode_lengths",data=episode_lengths)
		hf.create_dataset("episode_start",data=episode_start)
		hf.create_dataset("episode_total",data = [episode_total])
		hf.create_dataset("x_test",data=x_test, shape=x_test.shape)
		hf.create_dataset("y_test",data=y_test, shape=y_test.shape)
else:
	with h5py.File(f_file, 'r') as hf:
		x_train = hf['x_train'][:]
		y_train = hf['y_train'][:]
		episode_lengths = hf['episode_lengths'][:]
		episode_start = hf['episode_lengths'][:]
		episode_total = hf['episode_total'][:]
		episode_total = episode_total[0]
		x_test = hf['x_test'][:]
		y_test = hf['y_test'][:]


print("Size of:")
print("- Training-set:\t\t{}".format(len(y_train)))
print("- Test-set:\t\t{}".format(len(x_test)))
#print("- Validation-set:\t{}".format(len(data.valid.labels)))



def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))



def new_conv_layer(input,			  # The previous layer.
			   num_input_channels, # Num. channels in prev. layer.
			   filter_size,		# Width and height of each filter.
			   num_filters,		# Number of filters.
			   use_pooling=True):  # Use 2x2 max-pooling.

	# Shape of the filter-weights for the convolution.
	# This format is determined by the TensorFlow API.
	shape = [filter_size, filter_size, num_input_channels, num_filters]

	# Create new weights aka. filters with the given shape.
	weights = new_weights(shape=shape)

	# Create new biases, one for each filter.
	biases = new_biases(length=num_filters)

	# Create the TensorFlow operation for convolution.
	# Note the strides are set to 1 in all dimensions.
	# The first and last stride must always be 1,
	# because the first is for the image-number and
	# the last is for the input-channel.
	# But e.g. strides=[1, 2, 2, 1] would mean that the filter
	# is moved 2 pixels across the x- and y-axis of the image.
	# The padding is set to 'SAME' which means the input image
	# is padded with zeroes so the size of the output is the same.
	layer = tf.nn.conv2d(input=input,
					 filter=weights,
					 strides=[1, 1, 1, 1],
					 padding='SAME')

	# Add the biases to the results of the convolution.
	# A bias-value is added to each filter-channel.
	layer += biases

	# Use pooling to down-sample the image resolution?
	if use_pooling:
		# This is 2x2 max-pooling, which means that we
		# consider 2x2 windows and select the largest value
		# in each window. Then we move 2 pixels to the next window.
		layer = tf.nn.max_pool(value=layer,
							   ksize=[1, 2, 2, 1],
							   strides=[1, 2, 2, 1],
							   padding='SAME')

	# Rectified Linear Unit (ReLU).
	# It calculates max(x, 0) for each input pixel x.
	# This adds some non-linearity to the formula and allows us
	# to learn more complicated functions.
	layer = tf.nn.relu(layer)

	# Note that ReLU is normally executed before the pooling,
	# but since relu(max_pool(x)) == max_pool(relu(x)) we can
	# save 75% of the relu-operations by max-pooling first.

	# We return both the resulting layer and the filter-weights
	# because we will plot the weights later.
	return layer, weights

	

def flatten_layer(layer):
	# Get the shape of the input layer.
	layer_shape = layer.get_shape()

	# The shape of the input layer is assumed to be:
	# layer_shape == [num_images, img_height, img_width, num_channels]

	# The number of features is: img_height * img_width * num_channels
	# We can use a function from TensorFlow to calculate this.
	num_features = layer_shape[1:4].num_elements()

	# Reshape the layer to [num_images, num_features].
	# Note that we just set the size of the second dimension
	# to num_features and the size of the first dimension to -1
	# which means the size in that dimension is calculated
	# so the total size of the tensor is unchanged from the reshaping.
	layer_flat = tf.reshape(layer, [-1, num_features])

	# The shape of the flattened layer is now:
	# [num_images, img_height * img_width * num_channels]

	# Return both the flattened layer and the number of features.
	return layer_flat, num_features


def new_fc_layer(input,		  # The previous layer.
			 num_inputs,	 # Num. inputs from prev. layer.
			 num_outputs,	# Num. outputs.
			 use_relu=True): # Use Rectified Linear Unit (ReLU)?

	# Create new weights and biases.
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_biases(length=num_outputs)

	# Calculate the layer as the matrix multiplication of
	# the input and weights, and then add the bias-values.
	layer = tf.matmul(input, weights) + biases

	# Use ReLU?
	if use_relu:
		layer = tf.nn.relu(layer)

	return layer

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)



layer_conv1, weights_conv1 = \
new_conv_layer(input=x_image,
			   num_input_channels=num_channels,
			   filter_size=filter_size1,
			   num_filters=num_filters1,
			   use_pooling=True)
#print("now layer2 input")
#print(layer_conv1.get_shape())	 
layer_conv2, weights_conv2 = \
new_conv_layer(input=layer_conv1,
			   num_input_channels=num_filters1,
			   filter_size=filter_size2,
			   num_filters=num_filters2,
			   use_pooling=True)
#print("now layer3 input")
#print(layer_conv2.get_shape())	 
			   
layer_conv3, weights_conv3 = \
new_conv_layer(input=layer_conv2,
			   num_input_channels=num_filters2,
			   filter_size=filter_size3,
			   num_filters=num_filters3,
			   use_pooling=True)
#print("now layer flatten input")
#print(layer_conv3.get_shape())	 
		  
layer_flat, num_features = flatten_layer(layer_conv3)

layer_fc1 = new_fc_layer(input=layer_flat,
					 num_inputs=num_features,
					 num_outputs=fc_size,
					 use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
					 num_inputs=fc_size,
					 num_outputs=num_classes,
					 use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)

y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
													labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#session.run(tf.global_variables_initializer()) # for newer versions
session.run(tf.initialize_all_variables()) # for older versions
train_batch_size = batch_size

def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
	# Calculate the accuracy on the training-set.
	acc = session.run(accuracy, feed_dict=feed_dict_train)
	val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
	msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
	print(msg.format(epoch + 1, acc, val_acc, val_loss))



total_iterations = 0

def optimize(num_iterations):
	# Ensure we update the global variable rather than a local copy.
	global total_iterations

	best_val_loss = float("inf")

	for i in range(total_iterations,total_iterations + num_iterations):

		# Get a batch of training examples.
		# x_batch now holds a batch of images and
		# y_true_batch are the true labels for those images.
		#x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
		#x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)
		
		# Convert shape from [num examples, rows, columns, depth]
		# to [num examples, flattened image shape]
		#x_batch = x_batch.reshape(train_batch_size, img_size_flat)
		#x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)
		# Put the batch into a dict with the proper names
		# for placeholder variables in the TensorFlow graph.

		ind =  random.sample(range(0, x_train.shape[0]), batch_size)#random.sample(range(0, num), 1)[0]
		ind_v = random.sample(range(0, x_train.shape[0]), batch_size)
		#indx = episode_start[ind]
		ind_data = ind#list(range(indx,indx+episode_lengths[ind]))

		#print(ind, indx, len(ind_data))
		x_batch = x_train[ind_data,:,:,:]
		y_true_batch = y_train[ind_data,:]
		x_valid_batch = x_train[ind_v,:,:,:]
		y_valid_batch = y_train[ind_v,:]

		feed_dict_train = {x: x_batch,
						   y_true: y_true_batch}
		
		feed_dict_validate = {x: x_valid_batch,
							  y_true: y_valid_batch}

		# Run the optimizer using this batch of training data.
		# TensorFlow assigns the variables in feed_dict_train
		# to the placeholder variables and then runs the optimizer.
		session.run(optimizer, feed_dict=feed_dict_train)
		
		#print(x_batch.shape)

		# Print status at end of each epoch (defined as full pass through training dataset).
		if i % int(x_train.shape[0]/batch_size) == 0: 
			val_loss = session.run(cost, feed_dict=feed_dict_validate)
			epoch = int(i / int(x_train.shape[0]/batch_size))
			
			print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)
			print("Testing Accuracy "+str(epoch)+": ", session.run(accuracy, feed_dict={x: x_test,
                                      y_true: y_test})*100)

	# Update the total number of iterations performed.
	total_iterations += num_iterations

	
optimize(num_iterations=3000)
#print_validation_accuracy()