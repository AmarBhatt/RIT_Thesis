import numpy as np
import tflearn
import tensorflow as tf
from DataSetGenerator.maze_generator import *

def getData (location):
	data = "" #store data as file names s,a,s'
	return data

def preprocess(data):
	return _data

def train(net,data,labels,n_epoch=1000,batch_size=32,show_metric=True):
	model = tflearn.DNN(net)
	model.fit([data],[labels],n_epoch,batch_size,show_metric)
	return model


def evaluate(input_data, model):
	return model.predict(input_data)

def DAQN(X,Y,learning_rate=0.01):
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
	net = tflearn.input_data(shape=[None,100,100,1], placeholder = X, name="inputlayer") #CHECK IF THESE ARE THE RIGHT DIMENSIONS!

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

	def sse(y_true, y_pred):
		#sum of square error
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
		Y = tf.placeholder(shape=[1,1,5], dtype="float32", name='a')
		daqn,daqn_presoft = DAQN(X,Y,0.05)
		X2 = tf.placeholder(shape=[1,100,100,1], dtype="float32",name='s')
		Y2 = tf.placeholder(shape=[1,1,5], dtype="float32", name='r_prime')
		darn = DARN(X2,Y2,0.05)
	#darn = DARN(0.05,0.06)

	print(daqn)
	print(darn)
	x_data = np.random.rand(1,100,100,1)
	y_data = np.random.rand(1,5)

	with tf.Session(graph=graph) as sess:
		#sess = tf.Session(graph=graph)

		sess.run(tf.global_variables_initializer())	
		writer = tf.summary.FileWriter('DAQN_log',graph=sess.graph)
		sess.run(daqn,feed_dict={X : x_data, Y : y_data})
		sess.run(darn,feed_dict={X2 : x_data, Y2 : y_data})
		
		#sess.run(daqn_presoft,feed_dict={X : x_data, Y : y_data})
		#m = train(daqn,x_data,y_data,n_epoch=1000,batch_size=32,show_metric=True)
		#daqn_ps = tflearn.DNN(daqn_presoft,session=m.session)

	writer.flush()
	writer.close()		



#graph = tf.Graph()
#with graph.as_default():
	#Place Holder

f_data = "same"
num = 800
num2 = 1000


image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(0))

x_data = image_set
y_data = action_set

for i in range(1,num):
	image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(i))
	x_data = np.append(x_data,image_set,axis=0)
	y_data = np.append(y_data,action_set, axis=0)

print(x_data.shape)
print(y_data.shape)

image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(num))

x_test = image_set
y_test = action_set

for i in range(num+1,num2):
	image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(i))
	x_test = np.append(x_test,image_set,axis=0)
	y_test = np.append(y_test,action_set, axis=0)

print(x_test.shape)
print(y_test.shape)
#x_data = np.random.rand(2,100,100,1)
#y_data = np.random.rand(2,5)

with tf.Graph().as_default():
	with tf.Session() as sess:
		X = tf.placeholder(shape=[None,100,100,1], dtype="float32",name='s')
		Y = tf.placeholder(shape=[1,5], dtype="float32", name='a')
		daqn,daqn_presoft = DAQN(X,None,0.01)
		model = tflearn.DNN(daqn)
			
		model.fit(x_data,y_data,n_epoch=20,batch_size=1, validation_set = 0.2, show_metric=True)
#result = sess.run(daqn_presoft, feed_dict={X : x_data})



		prediction=tf.argmax(daqn,1)
		result = model.predict(x_test)
		final_result = np.zeros(shape=[y_test.shape[0]], dtype=np.uint8)
		pred_result = np.zeros(shape=[y_test.shape[0]], dtype=np.uint8)
		for i in range(len(result)):
			r = result[i]
			a = np.where(y_test[i] == y_test[i].max())
			r = r.index(max(r))
			pred_result[i] = r
			#print(r,a[0][0])
			final_result[i] = int(r) == int(a[0][0])

		#print(result)
		#print(final_result)
		print("Test results")
		print("0: "+str((pred_result==0).sum()) +", 1: "+str((pred_result==1).sum()) + ", 2: "+str((pred_result==2).sum()) + ", 3: "+str((pred_result==3).sum()) + ", 4: "+str((pred_result==4).sum()))
		print(np.mean(final_result))
		print(str(np.count_nonzero(final_result)) + '/' + str(y_test.shape[0]))

# with tf.Session(graph=graph) as sess:
# 	#sess = tf.Session(graph=graph)
# 	#tf.reset_default_graph()
	

# 	#sess.run(tf.global_variables_initializer())	
# 	writer = tf.summary.FileWriter('DAQN_log',graph=sess.graph)
# 	#sess.run(daqn,feed_dict={X : x_data, Y : y_data})
# 	#sess.run(darn,feed_dict={X2 : x_data, Y2 : y_data})
	
# 	#sess.run(daqn_presoft,feed_dict={X : x_data, Y : y_data})
# 	#m = train(daqn,x_data,y_data,n_epoch=1000,batch_size=32,show_metric=True)
# 	#daqn_ps = tflearn.DNN(daqn_presoft,session=m.session)

# writer.flush()
# writer.close()