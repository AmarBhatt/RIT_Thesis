import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMCell
import scipy.misc

class DRQN:

	def __init__(self,X,Y,num_actions,rnn_cell,learning_rate,pooling_bool = True, myScope="drqn"):

		h_size = 512
		self.conv1 = slim.convolution2d( \
			inputs=X,num_outputs=32,\
			kernel_size=[8,8],stride=[4,4],padding='VALID', \
			biases_initializer=None,scope=myScope+'_conv1')
		self.conv2 = slim.convolution2d( \
			inputs=self.conv1,num_outputs=64,\
			kernel_size=[4,4],stride=[2,2],padding='VALID', \
			biases_initializer=None,scope=myScope+'_conv2')
		self.conv3 = slim.convolution2d( \
			inputs=self.conv2,num_outputs=64,\
			kernel_size=[3,3],stride=[1,1],padding='VALID', \
			biases_initializer=None,scope=myScope+'_conv3')
		self.conv4 = slim.convolution2d( \
			inputs=self.conv3,num_outputs=512,\
			kernel_size=[7,7],stride=[1,1],padding='VALID', \
			biases_initializer=None,scope=myScope+'_conv4')
		
		self.trainLength = tf.placeholder(dtype=tf.int32)
		#We take the output from the final convolutional layer and send it to a recurrent layer.
		#The input must be reshaped into [batch x trace x units] for rnn processing, 
		#and then returned to [batch x units] when sent through the upper levles.
		self.batch_size = tf.placeholder(dtype=tf.int32)
		self.convFlat = tf.reshape(slim.flatten(self.conv4),[self.batch_size,self.trainLength,h_size])
		self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
		self.rnn,self.rnn_state = tf.nn.dynamic_rnn(\
				inputs=self.convFlat,cell=rnn_cell,dtype=tf.float32,initial_state=self.state_in,scope=myScope+'_rnn')
		self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])
		#The output from the recurrent player is then split into separate Value and Advantage streams
		self.streamA,self.streamV = tf.split(self.rnn,2,1)
		self.AW = tf.Variable(tf.random_normal([h_size//2,num_actions]))
		self.VW = tf.Variable(tf.random_normal([h_size//2,1]))
		self.Advantage = tf.matmul(self.streamA,self.AW)
		self.Value = tf.matmul(self.streamV,self.VW)
		
		self.salience = tf.gradients(self.Advantage,X)
		#Then combine them together to get our final Q-values.
		self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,reduction_indices=1,keep_dims=True))
		self.predict = tf.argmax(self.Qout,1)
		
		#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
		#self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
		#self.actions_onehot = tf.one_hot(self.actions,4,dtype=tf.float32)
		self.actions_onehot = tf.placeholder(shape=[None,num_actions],dtype=tf.float32)
		
		self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)
		
		self.td_error = tf.square(self.targetQ - self.Q)
		#self.td_error = tf.Print(self.td_error,[self.td_error],message="td_error: ")
		#In order to only propogate accurate gradients through the network, we will mask the first
		#half of the losses for each trace as per Lample & Chatlot 2016
		self.maskA = tf.zeros([self.batch_size,self.trainLength//2])
		self.maskB = tf.ones([self.batch_size,self.trainLength//2])
		self.mask = tf.concat([self.maskA,self.maskB],1)
		self.mask = tf.reshape(self.mask,[-1])
		self.loss = tf.reduce_mean(self.td_error * self.mask)
		#self.loss = tf.Print(self.loss,[self.loss],message="Loss: ")
		
		self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		self.updateModel = self.trainer.minimize(self.loss)
		

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
