import numpy as np
import tflearn
import tensorflow as tf

graph = tf.Graph()

with graph.as_default():

	#Place Holder
	X = tf.placeholder(shape=[1,83,83,1], dtype="float32",name='X')
	Y = tf.placeholder(shape=[1,3], dtype="float32")
	# Input
	net = tflearn.input_data(shape=[1,83,83,1],placeholder=X,name="inputlayer") #CHECK IF THESE ARE THE RIGHT DIMENSIONS!

	# layer 1
	net = tflearn.layers.conv.conv_2d(net,nb_filter=16,filter_size=8, strides=4, activation='relu',name="convlayer1")
	net = tflearn.layers.conv.max_pool_2d(net,kernel_size=4,name="maxpool4")
			
	# layer 2
	net = tflearn.layers.conv.conv_2d(net,nb_filter=32,filter_size=4, strides=4,
				activation='relu',name="convlayer2")
	net = tflearn.layers.conv.max_pool_2d(net,kernel_size=2,name="maxpool2")

	# layer 3
	net = tflearn.fully_connected(net,n_units=2048,activation='tanh', name="FC")

	# Output
	#self.net = tflearn.layers.estimator.regression(self.net,name="FCout")
	net = tflearn.fully_connected(net,n_units=256, name="FCout")
	net = tflearn.fully_connected(net,n_units=3, name="FCpresoft")
	net = tflearn.fully_connected(net,n_units=3, activation='softmax', name="FCpostsoft")

	def sse(y_true, y_pred):
		#sum of square error
		return tf.square(tf.subtract(y_true,y_pred)) #SHOULD THIS STILL BE SUMMED?

	# loss = tf.square(tf.subtract(net,Y)) #SHOULD THIS STILL BE SUMMED?

	net = tflearn.regression(net, optimizer = tflearn.AdaGrad(learning_rate=0.01), loss=sse)

	print(net)

	#sess = tf.Session(graph=graph)
	
x_data = np.random.rand(1,83,83,1)
y_data = np.random.rand(1,3)
with tf.Session(graph=graph) as sess:
	#sess = tf.Session(graph=graph)
	
	sess.run(tf.global_variables_initializer())	
	writer = tf.summary.FileWriter('DAQN_log', sess.graph)
	sess.run(net,feed_dict={X:x_data, Y:y_data})
writer.flush()
writer.close()

# import tensorflow as tf 
# a = tf.constant(2, name="a") 
# b = tf.constant(3, name="b") 
# x = tf.add(a, b, name="add") 
# with tf.Session() as sess: 
# 	writer = tf.summary.FileWriter("./graphs", sess.graph) 
# 	print (sess.run(x)) # >> 5 