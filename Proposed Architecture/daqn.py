import numpy as np
import tensorflow as tf

class DAQN:

	def __init__(self,X,Y,weights, biases):

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
		#net = tflearn.input_data(shape=[None,100,100,1], placeholder = X, name="inputlayer") #CHECK IF THESE ARE THE RIGHT DIMENSIONS!
		#X = tf.placeholder(shape=[None,100,100,1], dtype="float32",name='X')
		#Y = tf.placeholder(shape=[1,5], dtype="float32", name='Y')
		# layer 1
		#net = tflearn.layers.conv.conv_2d(net,nb_filter=16,filter_size=[8,8], strides=[1,4,4,1], padding="valid",activation='relu',name="convlayer1")
		conv1 = self.conv2d(X, weights['wc1'], biases['bc1'], strides = 4, padding="VALID")
		#net = tflearn.layers.conv.max_pool_2d(net,kernel_size=[1,4,4,1], strides=1, name="maxpool4")
		conv1 = self.maxpool2d(conv1,k=4,s=1)	
		# layer 2
		#net = tflearn.layers.conv.conv_2d(net,nb_filter=32,filter_size=[4,4], strides=[1,1,1,1],padding="valid",activation='relu',name="convlayer2")
		conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'], strides = 1, padding="VALID")
		#net = tflearn.layers.conv.max_pool_2d(net,kernel_size=[1,2,2,1], strides=2,name="maxpool2")
		conv2 = self.maxpool2d(conv2,k=2,s=2)	

		# layer 3
		#layer2_flatten = tflearn.flatten(net,name="flatten")#tf.reshape(net,[-1,8*8*32])
		fc1 = self.flatten(conv2, weights['wd1'].get_shape().as_list()[0])
		#net = tflearn.fully_connected(layer2_flatten,n_units=256,activation='tanh', name="FC")
		fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
		fc1 = tf.nn.tanh(fc1)
		# Output
		#self.net = tflearn.layers.estimator.regression(self.net,name="FCout")
		#net = tflearn.fully_connected(net,n_units=256, name="FCout")
		#layer_presoft = tflearn.fully_connected(net,n_units=5, name="FCpresoft")
		outpre = tf.add(tf.matmul(fc1,weights['out']),biases['out'])
		#net = tflearn.fully_connected(layer_presoft,n_units=5, activation='softmax', name="FCpostsoft")
		outpost = tf.nn.softmax(outpre,dim=-1)		

		self.outpre = outpre
		self.outpost = outpost

	def conv2d(self, x, W, b, strides=1, padding="SAME"):
    # Conv2D wrapper, with bias and relu activation
	    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
	    x = tf.nn.bias_add(x, b)
	    return tf.nn.relu(x)

	def maxpool2d(self, x, k=2, s=2):
	    # MaxPool2D wrapper
	    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME')

	def flatten(self, x, dim):
		#Flatten wrapper
		#dims = int(np.prod(x.get_shape().as_list()[1:])) #may need to splice [1:] because first x.get_shape().as_list() is None
		x = tf.reshape(x, [-1,dim])
		return x

def sse(y_pred, y_true):
	#sum of square error
	#y_true = tf.Print(y_true, [y_true], message="y_true is: ")
	#y_pred = tf.Print(y_pred, [y_pred], message="y_pred is: ")
	loss = tf.square(tf.subtract(y_true,y_pred))
	loss.set_shape([1,5])
	return tf.reduce_sum(loss) #SHOULD THIS STILL BE SUMMED?

def generateNetworkStructure():
	
	n_classes = 5 #5 actions
	learning_rate = 0.001
	

	graph = tf.Graph()
	with graph.as_default():

		# Store layers weight & bias
		weights = {
		    # 8x8 conv, 1 input, 16 outputs
		    'wc1': tf.Variable(tf.random_normal([8, 8, 1, 16])),
		    # 4x4 conv, 16 inputs, 32 outputs
		    'wc2': tf.Variable(tf.random_normal([4, 4, 16, 32])),
		    # fully connected, 7*7*64 inputs, 1024 outputs
		    'wd1': tf.Variable(tf.random_normal([11*11*32, 256])),
		    # 1024 inputs, 10 outputs (class prediction)
		    'out': tf.Variable(tf.random_normal([256, n_classes]))
		}

		biases = {
		    'bc1': tf.Variable(tf.random_normal([16])),
		    'bc2': tf.Variable(tf.random_normal([32])),
		    'bd1': tf.Variable(tf.random_normal([256])),
		    'out': tf.Variable(tf.random_normal([n_classes]))
		}	

	
		#Place Holder
		X = tf.placeholder(shape=[1,100,100,1], dtype="float32",name='s')
		Y = tf.placeholder(shape=[1,5], dtype="float32", name='a')
		net = DAQN(X,Y, weights, biases)
		daqn = net.outpost
		daqn_presoft = net.outpre
		
		# Define Loss and optimizer
		cost = sse(daqn,Y)
		optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

		# Evaluate model
		correct_pred = tf.equal(tf.argmax(daqn, 0), tf.argmax(Y, 0)) #1 instead of 0?
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		#Initialize variables
		init = tf.global_variables_initializer()
	#print(daqn)
	#print(darn)
	x_data = np.random.rand(1,100,100,1)
	y_data = np.random.rand(1,5)

	with tf.Session(graph=graph) as sess:

		sess.run(init)	
		writer = tf.summary.FileWriter('DAQN_log_test',graph=sess.graph)
		sess.run(optimizer,feed_dict={X : x_data, Y : y_data})


	writer.flush()
	writer.close()		

#generateNetworkStructure()