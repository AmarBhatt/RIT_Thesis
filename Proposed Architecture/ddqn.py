import numpy as np
import tensorflow as tf

class DDQN:

	def __init__(self,X,Y,num_actions,learning_rate=0.0001,pooling_bool = True,h_size=512):

		self.conv1 = tf.contrib.layers.convolution2d( \
			inputs=X,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
		self.conv2 = tf.contrib.layers.convolution2d( \
			inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
		self.conv3 = tf.contrib.layers.convolution2d( \
			inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
		self.conv4 = tf.contrib.layers.convolution2d( \
			inputs=self.conv3,num_outputs=h_size,kernel_size=[7,7],stride=[1,1],padding='VALID', biases_initializer=None)
		
		#We take the output from the final convolutional layer and split it into separate advantage and value streams.
		self.streamAC,self.streamVC = tf.split(self.conv4,2,3) #self.streamAC,self.streamVC = tf.split(3,2,self.conv4)
		self.streamA = tf.contrib.layers.flatten(self.streamAC)
		self.streamV = tf.contrib.layers.flatten(self.streamVC)
		self.AW = tf.Variable(tf.random_normal([h_size//2,num_actions]))
		self.VW = tf.Variable(tf.random_normal([h_size//2,1]))
		self.Advantage = tf.matmul(self.streamA,self.AW)
		self.Value = tf.matmul(self.streamV,self.VW)

		 #Then combine them together to get our final Q-values.
		self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,reduction_indices=1,keep_dims=True))
		self.predict = tf.argmax(self.Qout,1)
		
		#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
		self.actions_onehot = tf.placeholder(shape=[None,num_actions],dtype=tf.float32)
		#self.actions_onehot = tf.one_hot(self.actions,env.actions,dtype=tf.float32)
		
		self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)
		
		self.td_error = tf.square(self.targetQ - self.Q)
		self.loss = tf.reduce_mean(self.td_error)
		self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate) #0.0001
		self.updateModel = self.trainer.minimize(self.loss)
		
		
	def conv2d(self, input,			  # The previous layer.
			   num_input_channels, # Num. channels in prev. layer.
			   filter_size,		# Width and height of each filter.
			   num_filters,		# Number of filters.
			   use_pooling=True,  # Use max-pooling. 
			   pooling = 2,		# kernel size
			   stride=1, 
			   pool_stride = 1,
			   padding="SAME",
			   pool_pad = "SAME"):
		# Conv2D wrapper, with bias and relu activation

		# Shape of the filter-weights for the convolution.
		# This format is determined by the TensorFlow API.
		shape = [filter_size, filter_size, num_input_channels, num_filters]
		# Create new weights aka. filters with the given shape.
		weights = self.new_weights(shape=shape)

		# Create new biases, one for each filter.
		biases = self.new_biases(length=num_filters)

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
						 strides=[1, stride, stride, 1],
						 padding=padding)

		# Add the biases to the results of the convolution.
		# A bias-value is added to each filter-channel.
		layer += biases

		# Use pooling to down-sample the image resolution?
		if use_pooling:
			# This is 2x2 max-pooling, which means that we
			# consider 2x2 windows and select the largest value
			# in each window. Then we move 2 pixels to the next window.
			layer = tf.nn.max_pool(value=layer,
								   ksize=[1, pooling, pooling, 1],
								   strides=[1, pool_stride, pool_stride, 1],
								   padding=pool_pad)

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

	def maxpool2d(self, x, k=2, s=2):
		# MaxPool2D wrapper
		return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME')

	def flatten(self, x, dim):
		#Flatten wrapper
		#dims = int(np.prod(x.get_shape().as_list()[1:])) #may need to splice [1:] because first x.get_shape().as_list() is None
		x = tf.reshape(x, [-1,dim])
		return x

	def flatten_layer(self,layer):
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

	def fc_layer(self,input,		  # The previous layer.
			 num_inputs,	 # Num. inputs from prev. layer.
			 num_outputs,	# Num. outputs.
			 activation=""): # Use Rectified Linear Unit (ReLU)?

		# Create new weights and biases.
		weights = self.new_weights(shape=[num_inputs, num_outputs])
		biases = self.new_biases(length=num_outputs)

		# Calculate the layer as the matrix multiplication of
		# the input and weights, and then add the bias-values.
		layer = tf.matmul(input, weights) + biases

		# Use ReLU?
		if activation == "relu":
			layer = tf.nn.relu(layer)
		elif activation == "tanh":
			layer = tf.nn.tanh(layer)

		return layer

	def new_weights(self,shape):
		return tf.Variable(tf.random_normal(shape))

	def new_biases(self,length):
		return tf.Variable(tf.constant(0.05, shape=[length]))

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

# Update target network with primary network
def updateTargetGraph(tfVars,tau):
	total_vars = len(tfVars)
	op_holder = []
	for idx,var in enumerate(tfVars[0:total_vars//2]):
		op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
	return op_holder

def updateTarget(op_holder,sess):
	for op in op_holder:
		sess.run(op)
