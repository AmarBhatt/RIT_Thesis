import numpy as np
import tflearn
import tensorflow as tf
from DataSetGenerator.maze_generator import *
from daqn import DAQN, sse
#import alexnet
import sys as sys
from PIL import Image
import os.path
import h5py
#Load necessary libraries
import tensorflow.contrib.slim as slim
#import input_data
import matplotlib.pyplot as plt
#%matplotlib inline

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


def getData (data_loc,location,data_size,num_train,num_test):
	f_data = location
	f_file = data_loc

	num = num_train
	num2 = num_train + num_test

	if not os.path.isfile(f_file):
		print("Premade dataset for "+f_file+" does not exist. Creating new data.")
		episode_lengths = []
		episode_start = []
		episode_total = 0
		image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(0),data_size)

		x_train = image_set
		y_train = action_set

		episode_lengths.append(action_set.shape[0])
		episode_start.append(episode_total)
		episode_total += image_set.shape[0]
		for i in range(1,num):
			if (i%(num//10) == 0):
				print(i)
			image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(i),data_size)
			x_train = np.append(x_train,image_set,axis=0)
			y_train = np.append(y_train,action_set, axis=0)
			episode_lengths.append(action_set.shape[0])
			episode_start.append(episode_total)
			episode_total += image_set.shape[0]
		
		print("Training set size - X: "+str(x_train.shape)+", Y: "+str(y_train.shape))

		image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(num),data_size)

		x_test = image_set
		y_test = action_set

		for i in range(num+1,num2):
			image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(i),data_size)
			x_test = np.append(x_test,image_set,axis=0)
			y_test = np.append(y_test,action_set, axis=0)

		print("Testing set size - X: "+str(x_test.shape)+", Y: "+str(y_test.shape))
		
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
	return x_train, y_train, episode_lengths, episode_start, episode_total, episode_total, x_test, y_test 


def train(net,data,labels,n_epoch=1000,batch_size=32,show_metric=True):
	model = tflearn.DNN(net)
	model.fit([data],[labels],n_epoch,batch_size,show_metric)
	return model


def evaluate(input_data, model):
	return model.predict(input_data)

def _DAQN(X,Y,shape,learning_rate=0.01):
	"""
	Input: 83x83x? (? = 1-4)

	1st layer (CONV): activation - TANH
		16 feature maps (19x19)
		16 filters (8x8) (stride 4)
		4x4 max-pooling

	2nd layer (CONV): activation - TANH
		32 feature maps (8x8)
		32 filters (4x4)
		2x2 max-pooling

	3rd layer (FC):
		input: 2048
		output: 256

	Output layer (FC):
		input: 256
		output: 3 (actions)
			soft-max non-linearity

	cost = squared sum of the difference of q(s,a) from network (so array of 3 values) and q*(s,a) from expert (1-hot encoded array of real action chosen)
	Learning rate is done with AdaGrad

	Trained with: expert trajectories (s,a)
	Hyperparameters: gamma (discount rate), nu (learning rate)
	"""
	# graph = tf.Graph()
	# with graph.as_default():

		
	# Input
	net = tflearn.input_data(shape=[None,shape,shape,1], placeholder = X, name="inputlayer") #CHECK IF THESE ARE THE RIGHT DIMENSIONS!

	# layer 1
	net = tflearn.layers.conv.conv_2d(net,nb_filter=16,filter_size=[8,8], strides=[1,4,4,1], padding="valid",activation='relu',name="convlayer1")
	net = tflearn.layers.conv.max_pool_2d(net,kernel_size=[1,4,4,1], strides=[1,1,1,1], name="maxpool4")
			
	# layer 2
	net = tflearn.layers.conv.conv_2d(net,nb_filter=32,filter_size=[4,4], strides=[1,1,1,1],padding="valid",
				activation='relu',name="convlayer2")
	net = tflearn.layers.conv.max_pool_2d(net,kernel_size=[1,2,2,1], strides=2,name="maxpool2")

	# layer 3
	layer2_flatten = tflearn.flatten(net,name="flatten")#tf.reshape(net,[-1,8*8*32])
	net = tflearn.fully_connected(layer2_flatten,n_units=256,activation='tanh', name="FC")

	# Output
	#self.net = tflearn.layers.estimator.regression(self.net,name="FCout")
	#net = tflearn.fully_connected(net,n_units=256, name="FCout")
	layer_presoft = tflearn.fully_connected(net,n_units=5, name="FCpresoft")
	net = tflearn.fully_connected(layer_presoft,n_units=5, activation='softmax', name="FCpostsoft")

	def sse(y_pred, y_true):
		#sum of square error
		#y_true = tf.Print(y_true, [y_true], message="y_true is: ")
		#y_pred = tf.Print(y_pred, [y_pred], message="y_pred is: ")
		loss = tf.square(tf.subtract(y_true,y_pred))
		loss.set_shape([1,5])
		return tf.reduce_sum(loss) #SHOULD THIS STILL BE SUMMED?

	#loss = tf.square(tf.subtract(net,Y)) #SHOULD THIS STILL BE SUMMED?

	net = tflearn.regression(net, optimizer = tflearn.AdaGrad(learning_rate=learning_rate), loss=sse)

	#print(net.get_shape())

	return net, layer_presoft

def DARN(X,Y,learning_rate=0.01):
	"""
	Input: 83x83x? (? = 1-4)

	1st layer (CONV): activation - TANH
		16 feature maps (19x19)
		16 filters (8x8) (stride 4)
		4x4 max-pooling

	2nd layer (CONV): activation - TANH
		32 feature maps (8x8)
		32 filters (4x4)
		2x2 max-pooling

	3rd layer (FC):
		input: 2048
		output: 256

	Output layer (FC):
		input: 256
		output: 3 (actions)
			soft-max non-linearity

	cost = squared sum of the difference of q(s,a) from network (so array of 3 values) and q*(s,a) from expert (1-hot encoded array of real action chosen)
	Learning rate is done with AdaGrad

	Trained with: expert trajectories (s,a)
	Hyperparameters: gamma (discount rate), nu (learning rate)
	"""
	# graph = tf.Graph()
	# with graph.as_default():

		
	# Input
	net = tflearn.input_data(shape=[None,100,100,1],placeholder=X,name="inputlayer") #CHECK IF THESE ARE THE RIGHT DIMENSIONS!

	# layer 1
	net = tflearn.layers.conv.conv_2d(net,nb_filter=16,filter_size=[8,8], strides=[1,4,4,1], padding="valid",activation='relu',name="convlayer1")
	net = tflearn.layers.conv.max_pool_2d(net,kernel_size=[1,4,4,1], strides=1, name="maxpool4")
			
	# layer 2
	net = tflearn.layers.conv.conv_2d(net,nb_filter=32,filter_size=[4,4], strides=[1,1,1,1],padding="valid",
				activation='relu',name="convlayer2")
	net = tflearn.layers.conv.max_pool_2d(net,kernel_size=[1,2,2,1], strides=2,name="maxpool2")

	# layer 3
	layer2_flatten = tflearn.flatten(net,name="flatten")#tf.reshape(net,[-1,8*8*32])
	net = tflearn.fully_connected(layer2_flatten,n_units=256,activation='tanh', name="FC")

	# Output
	#self.net = tflearn.layers.estimator.regression(self.net,name="FCout")
	#net = tflearn.fully_connected(net,n_units=256, name="FCout")
	layer_presoft = tflearn.fully_connected(net,n_units=5, name="FCpresoft")
	net = tflearn.fully_connected(layer_presoft,n_units=5, activation='softmax', name="FCpostsoft")

	#def l2(y_true, y_pred):
		# y_true = q(s,a)[before softmax] - gamma*max(q(s',a')[before softmax])
		#These q(s,a) and q(s',a') come from trained DQN

	#	return tflearn.losses.L2(tf.subtract(y_pred,y_true))

	loss = tflearn.losses.L2(tf.subtract(net,Y))

	net = tflearn.regression(net, optimizer = tflearn.AdaGrad(learning_rate=learning_rate), loss=loss)

	return net


def generateNetworkStructure():
	graph = tf.Graph()
	with graph.as_default():
		#Place Holder
		X = tf.placeholder(shape=[1,100,100,1], dtype="float32",name='s')
		Y = tf.placeholder(shape=[1,5], dtype="float32", name='a')
		daqn,daqn_presoft = DAQN(X,Y,0.05)
		#X2 = tf.placeholder(shape=[1,100,100,1], dtype="float32",name='s')
		#Y2 = tf.placeholder(shape=[1,5], dtype="float32", name='r_prime')
		#darn = DARN(X2,Y2,0.05)
	#darn = DARN(0.05,0.06)

	print(daqn)
	#print(darn)
	x_data = np.random.rand(1,100,100,1)
	y_data = np.random.rand(1,5)

	with tf.Session(graph=graph) as sess:
		#sess = tf.Session(graph=graph)

		sess.run(tf.global_variables_initializer())	
		writer = tf.summary.FileWriter('DAQN_log',graph=sess.graph)
		sess.run(daqn,feed_dict={X : x_data, Y : y_data})
		#sess.run(darn,feed_dict={X2 : x_data, Y2 : y_data})
		
		#sess.run(daqn_presoft,feed_dict={X : x_data, Y : y_data})
		#m = train(daqn,x_data,y_data,n_epoch=1000,batch_size=32,show_metric=True)
		#daqn_ps = tflearn.DNN(daqn_presoft,session=m.session)

	writer.flush()
	writer.close()		

data_loc = "same_1000.h5"
location = "same"
data_size = 83
num_train = 800
num_test = 200

netName = "daqn"
log = netName + "_log"

num_epochs = 100 #20
episodes = 50
batch_size = 1
test_batch_size = 50

n_classes = 5 #5 actions
learning_rate = 0.001

x_train, y_train, episode_lengths, episode_start, episode_total, episode_total, x_test, y_test = getData(data_loc,location,data_size,num_train,num_test)

graph = tf.Graph()
with graph.as_default():
	#tf.reset_default_graph()
	#Initialize variables
	
	if(netName == "daqn"):	
		# Store layers weight & bias
		# weights = {
		#     # 8x8 conv, 1 input, 16 outputs
		#     'wc1': tf.Variable(tf.random_normal([8, 8, 1, 16])),
		#     # 4x4 conv, 16 inputs, 32 outputs
		#     'wc2': tf.Variable(tf.random_normal([4, 4, 16, 32])),
		#     # fully connected, 7*7*64 inputs, 1024 outputs
		#     'wd1': tf.Variable(tf.random_normal([11*11*32, 256])), #was [21*21*32, 256] without Max Pool
		#     # 1024 inputs, 10 outputs (class prediction)
		#     'out': tf.Variable(tf.random_normal([256, n_classes]))
		# }

		# biases = {
		#     'bc1': tf.Variable(tf.random_normal([16])),
		#     'bc2': tf.Variable(tf.random_normal([32])),
		#     'bd1': tf.Variable(tf.random_normal([256])),
		#     'out': tf.Variable(tf.random_normal([n_classes]))
		# }
		#Place Holder
		X = tf.placeholder(shape=[None,data_size,data_size,1], dtype="float32",name='s')
		X = tf.reshape(X, [-1, data_size, data_size, 1])

		Y = tf.placeholder(shape=[None,n_classes], dtype="float32", name='a')
		net = DAQN(X,Y)
		daqn = net.outpost
		daqn_presoft = net.outpre
		
		# Define Loss and optimizer
		#cost = sse(daqn,Y)
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=daqn_presoft,
													labels=Y)
		cost = tf.reduce_mean(cross_entropy)

		optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
		update = optimizer.minimize(cost)
		
		# Evaluate model
		pred = tf.argmax(daqn, 1)
		pred = tf.Print(pred,[pred],message="prediction is: ")
		true = tf.argmax(Y, 1)
		true = tf.Print(true,[true],message="truth is: ")


		correct_pred = tf.equal(pred, true) #1 instead of 0?
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	init = tf.global_variables_initializer()

with tf.Session(graph=graph) as sess:
	sess.run(init)	
	writer = tf.summary.FileWriter(log,graph=sess.graph)
	
	for epoch in range(num_epochs):
			print("Epoch: "+str(epoch))
			for ep in range(episodes):

				ind =  random.sample(range(0, x_train.shape[0]), batch_size)#random.sample(range(0, num), 1)[0]
				#indx = episode_start[ind]
				ind_data = ind#list(range(indx,indx+episode_lengths[ind]))

				#print(ind, indx, len(ind_data))
				x_data = x_train[ind_data,:,:,:]
				y_data = y_train[ind_data,:]
				
				# im = Image.fromarray(x_data[0].squeeze(axis=2),'L')
				# im.show()
				# input("show image")
				sess.run(update,feed_dict={X : x_data, Y : y_data})

				# Display loss and accuracy
				loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data,
                                                              Y: y_data})
				print("Epoch= "+str(epoch)+", Episode= " + str(ep) + ", Minibatch Loss= " + \
	                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
	                  "{:.5f}".format(acc))
			
			ind =  random.sample(range(0, x_test.shape[0]), test_batch_size)
			x_test_batch = x_test[ind,:,:,:]
			y_test_true_batch = y_test[ind,:]	

			print("Testing Accuracy "+str(epoch)+": ", sess.run(accuracy, feed_dict={X: x_test_batch,
                                      Y: y_test_true_batch}))

writer.flush()
writer.close()