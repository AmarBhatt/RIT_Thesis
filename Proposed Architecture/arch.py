import numpy as np
np.set_printoptions(threshold=np.nan)
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


def getData (data_loc,location,rew_location,data_size,num_train,num_test,num_reward,skip_goal = None,normalize=1):
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

		x_train = np.empty(shape=[200000,data_size,data_size,1])
		y_train = np.empty(shape=[200000,len(range(0,5)[0:skip_goal])])

		data_index = 0

		im_set = image_set[0:skip_goal,:,:,:]
		a_set = action_set[0:skip_goal,0:skip_goal]

		x_train[data_index:data_index+im_set.shape[0],:,:,:] = np.divide(im_set,normalize)
		y_train[data_index:data_index+a_set.shape[0],:] = a_set

		data_index += a_set.shape[0]

		episode_lengths.append(a_set.shape[0])
		episode_start.append(episode_total)
		episode_total += im_set.shape[0]
		for i in range(1,num):
			#if (i%(num//10) == 0):
			print(i,data_index)
			image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(i),data_size)
			image_set = np.divide(image_set,normalize)
			
			im_set = image_set[0:skip_goal,:,:,:]
			a_set = action_set[0:skip_goal,0:skip_goal]

			#x_train = np.append(x_train,im_set,axis=0)
			#y_train = np.append(y_train,a_set, axis=0)

			x_train[data_index:data_index+im_set.shape[0],:,:,:] = im_set
			y_train[data_index:data_index+a_set.shape[0],:] = a_set

			data_index += a_set.shape[0]

			episode_lengths.append(a_set.shape[0])
			episode_start.append(episode_total)
			episode_total += im_set.shape[0]
		#print(np.array_str(image_set[0,:,:,:],75,3))
		#input("train pause")
		x_train.resize(data_index,data_size,data_size,1)
		y_train.resize(data_index,len(range(0,5)[0:skip_goal]))
		print("Training set size - X: "+str(x_train.shape)+", Y: "+str(y_train.shape))
		image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(i),data_size)
		
		x_test = np.empty(shape=[100000,data_size,data_size,1])
		y_test = np.empty(shape=[100000,len(range(0,5)[0:skip_goal])])

		data_index = 0

		im_set = image_set[0:skip_goal,:,:,:]
		a_set = action_set[0:skip_goal,0:skip_goal]
		#x_test = np.divide(im_set,normalize)
		#y_test = a_set

		x_test[data_index:data_index+im_set.shape[0],:,:,:] = np.divide(im_set,normalize)
		y_test[data_index:data_index+a_set.shape[0],:] = a_set

		data_index += a_set.shape[0]

		for i in range(num+1,num2):
			print(i,data_index)
			image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(i),data_size)
			image_set = np.divide(image_set,normalize)
			im_set = image_set[0:skip_goal,:,:,:]
			a_set = action_set[0:skip_goal,0:skip_goal]
			#x_test = np.append(x_test,im_set,axis=0)
			#y_test = np.append(y_test,a_set, axis=0)

			x_test[data_index:data_index+im_set.shape[0],:,:,:] = im_set
			y_test[data_index:data_index+a_set.shape[0],:] = a_set

			data_index += a_set.shape[0]
		
		#print(image_set[0,:,:,:])
		#input("test pause")
		x_test.resize(data_index,data_size,data_size,1)
		y_test.resize(data_index,len(range(0,5)[0:skip_goal]))
		print("Testing set size - X: "+str(x_test.shape)+", Y: "+str(y_test.shape))

		image_set,action_set = processGIF('DataSetGenerator/random_data/'+rew_location+"/"+str(0),data_size)
		
		state = np.empty(shape=[100000,data_size,data_size,1])
		action = np.empty(shape=[100000,len(range(0,5)[0:skip_goal])])
		state_prime = np.empty(shape=[100000,data_size,data_size,1])
		action_prime = np.empty(shape=[100000,len(range(0,5)[0:skip_goal])])

		data_index = 0 

		im_set = np.divide(image_set,normalize)
		a_set = action_set[:,0:skip_goal]
		
		state[data_index,:,:,:] = im_set[0,:,:,:]
		action[data_index,:] = a_set[0,:]
		state_prime[data_index,:,:,:] = im_set[1,:,:,:]
		action_prime[data_index,:] = a_set[1,:]

		data_index += 1
		#state = [im_set[0,:,:,:]]
		#action = [a_set[0,:]]
		#state_prime = [im_set[1,:,:,:]]
		#action_prime = [a_set[1,:]]

		for i in range(1,num_reward):
			#if (i%(num_reward//10) == 0):
			print(i,data_index)
			image_set,action_set = processGIF('DataSetGenerator/random_data/'+rew_location+"/"+str(i),data_size)
			im_set = np.divide(image_set,normalize)
			a_set = action_set[:,0:skip_goal]


			if(skip_goal == -1):
				if(action_set[0,4] == 0 and action_set[1,4] == 0):
					# state = np.append(state,[im_set[0,:,:,:]],axis=0)
					# action = np.append(action,[a_set[0,:]],axis=0)
					# state_prime = np.append(state_prime,[im_set[1,:,:,:]],axis=0)
					# action_prime = np.append(action_prime,[a_set[1,:]],axis=0)
					state[data_index,:,:,:] = im_set[0,:,:,:]
					action[data_index,:] = a_set[0,:]
					state_prime[data_index,:,:,:] = im_set[1,:,:,:]
					action_prime[data_index,:] = a_set[1,:]

					data_index += 1
			else:
				# state = np.append(state,[image_set[0,:,:,:]],axis=0)
				# action = np.append(action,[a_set[0,:]],axis=0)
				# state_prime = np.append(state_prime,[image_set[1,:,:,:]],axis=0)
				# action_prime = np.append(action_prime,[a_set[1,:]],axis=0)
				state[data_index,:,:,:] = im_set[0,:,:,:]
				action[data_index,:] = a_set[0,:]
				state_prime[data_index,:,:,:] = im_set[1,:,:,:]
				action_prime[data_index,:] = a_set[1,:]

				data_index += 1
			
		#print(image_set[0,:,:,:])
		#input("reward pause")
		state.resize(data_index,data_size,data_size,1)
		action.resize(data_index,len(range(0,5)[0:skip_goal]))
		state_prime.resize(data_index,data_size,data_size,1)
		action_prime.resize(data_index,len(range(0,5)[0:skip_goal]))
		print("Reward set size - State: "+str(state.shape)+", Action: "+str(action.shape)+", State': "+str(state_prime.shape)+", Action': "+str(action_prime.shape))


		
		with h5py.File(f_file,'w') as hf:
			hf.create_dataset("x_train",data=x_train, shape = x_train.shape)
			hf.create_dataset("y_train",data=y_train, shape = y_train.shape)
			hf.create_dataset("episode_lengths",data=episode_lengths)
			hf.create_dataset("episode_start",data=episode_start)
			hf.create_dataset("episode_total",data = [episode_total])
			hf.create_dataset("x_test",data=x_test, shape=x_test.shape)
			hf.create_dataset("y_test",data=y_test, shape=y_test.shape)
			hf.create_dataset("state",data=state, shape=state.shape)
			hf.create_dataset("action",data=action, shape=action.shape)
			hf.create_dataset("state_prime",data=state_prime, shape=state_prime.shape)
			hf.create_dataset("action_prime",data=action_prime, shape=action_prime.shape)
	else:
		with h5py.File(f_file, 'r') as hf:
			x_train = hf['x_train'][:]
			y_train = hf['y_train'][:]
			episode_lengths = hf['episode_lengths'][:]
			episode_start = hf['episode_start'][:]
			episode_total = hf['episode_total'][:]
			episode_total = episode_total[0]
			x_test = hf['x_test'][:]
			y_test = hf['y_test'][:]
			state = hf['state'][:]
			action = hf['action'][:]
			state_prime = hf['state_prime'][:]
			action_prime = hf['action_prime'][:]
			print("Training set size - X: "+str(x_train.shape)+", Y: "+str(y_train.shape))
			print("Testing set size - X: "+str(x_test.shape)+", Y: "+str(y_test.shape))
			print("Reward set size - State: "+str(state.shape)+", Action: "+str(action.shape)+", State': "+str(state_prime.shape)+", Action': "+str(action_prime.shape))
	return x_train, y_train, episode_lengths, episode_start, episode_total, episode_total, x_test, y_test,state,action,state_prime,action_prime


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

