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

from arch import getData

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression



data_loc = "same_1000.h5"
location = "same"
same = True
data_size = 83
num_train = 800
num_test = 200
path = os.path.dirname(os.path.realpath(__file__))
netName = "daqn"
log = netName + "_log"
daqn_model_path = path+'\saved-models\daqn\daqn.ckpt'
darn_model_path = path+'\saved-models\darn\darn.ckpt'


num_epochs = 10 #20
episodes = 5
num_epochs_darn = 10 #20
episodes_darn = 5
batch_size = 5
batch_size_darn = 50
test_batch_size = 50

n_classes = 5 #5 actions
learning_rate = 0.01
gamma = 0.01

restore_epoch = num_epochs-1

x_train, y_train, episode_lengths, episode_start, episode_total, episode_total, x_test, y_test = getData(data_loc,location,data_size,num_train,num_test)


graph_daqn = tf.Graph()
with graph_daqn.as_default():
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

	
	init_daqn = tf.global_variables_initializer()
	saver_daqn = tf.train.Saver()

with tf.Session(graph=graph_daqn) as sess:
	sess.run(init_daqn)	
	writer = tf.summary.FileWriter(log,graph=sess.graph)
	
	for epoch in range(num_epochs):
			#print("Epoch: "+str(epoch))
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
			if(epoch%(num_epochs//10) == 0):
				daqn_save_path = saver_daqn.save(sess, daqn_model_path)#, global_step=epoch)
				print("Model saved in file: %s" % daqn_save_path)

			ind =  random.sample(range(0, x_test.shape[0]), test_batch_size)
			x_test_batch = x_test[ind,:,:,:]
			y_test_true_batch = y_test[ind,:]	

			print("Testing Accuracy "+str(epoch)+": ", sess.run(accuracy, feed_dict={X: x_test_batch,
                                      Y: y_test_true_batch}))

	#saver.save(sess, 'saved-models/daqn')
writer.flush()
writer.close()


graph_darn = tf.Graph()
with graph_darn.as_default():
	#DARN
	X_darn = tf.placeholder(shape=[None,data_size,data_size,1], dtype="float32",name='X_darn')
	X_darn = tf.reshape(X_darn, [-1, data_size, data_size, 1])

	Y_darn = tf.placeholder(shape=[None,5], dtype="float32", name='Y_darn')

	action_true = tf.placeholder(shape=[None,5], dtype="float32", name='action_true')

	net = DAQN(X_darn,Y_darn)
	darn = net.outpost
	darn_presoft = net.outpre

	#cost_darn = tf.norm(tf.subtract(darn_presoft,Y_darn), ord='euclidean')
	cost_darn_0 = tf.square(tf.subtract(tf.multiply(darn_presoft,action_true),tf.multiply(Y_darn,action_true)))
	cost_darn_0 = tf.reduce_max(cost_darn_0,axis=1)
	#cost_darn_0 = tf.Print(cost_darn_0,[cost_darn_0],message="cost is: ")
	cost_darn = tf.reduce_mean(cost_darn_0)
	#cost_darn = tf.Print(cost_darn,[cost_darn],message="cost* is: ")

	optimizer_darn = tf.train.AdagradOptimizer(learning_rate=learning_rate)
	update_darn = optimizer_darn.minimize(cost_darn)
	
	# Evaluate model
	pred_darn = tf.multiply(darn_presoft,action_true)
	pred_darn = tf.Print(pred_darn,[pred_darn],message="prediction is: ")
	true_darn = Y_darn
	true_darn = tf.Print(true_darn,[true_darn],message="truth is: ")

	correct_pred_darn = tf.subtract(pred_darn, true_darn) #1 instead of 0?
	accuracy_darn = tf.cast(tf.reduce_max(tf.abs(correct_pred_darn)), tf.float32)

	init_darn = tf.global_variables_initializer()
	saver_darn = tf.train.Saver()

sess_daqn = tf.Session(graph=graph_daqn)
sess_daqn.run(init_daqn)
saver_daqn.restore(sess_daqn,daqn_model_path)

sess_darn = tf.Session(graph=graph_darn)
sess_darn.run(init_darn)

writer_darn = tf.summary.FileWriter('DARN_log',graph=sess.graph)

for epoch in range(num_epochs_darn):
	for ep in range(episodes_darn):


		episode =  random.sample(range(0, len(episode_lengths)), batch_size_darn)
		ep_start = episode_start[episode]
		ep_end = ep_start + episode_lengths[episode]

		state_ind = []
		for i in range(ep_end.size):
			state_ind.append(random.sample(range(ep_start[i], ep_end[i]-1), 1)[0])

		state_ind = np.array(state_ind)
		state = x_train[state_ind,:,:,:]
		action = y_train[state_ind,:]
		state_p = x_train[state_ind+1,:,:,:]

		
		Q = sess_daqn.run(daqn_presoft,feed_dict={X: state,Y: action})
		#print(Q, action,np.argmax(action))
		Q = Q[:,np.argmax(action)]
		#print(Q)
		Q_p = sess_daqn.run(daqn_presoft,feed_dict={X: state_p,Y: action})
		#print(Q_p)
		Q_p = np.amax(Q_p)
		#print(Q_p)
		r_hat = action
		r_hat[:,np.argmax(action)] = Q-(gamma*Q_p)

		#r_hat = [[r_hat]]

		sess_darn.run(update_darn,feed_dict={X_darn: state, Y_darn: r_hat, action_true: action})

		# Display loss and accuracy
		loss, acc = sess_darn.run([cost_darn, accuracy_darn], feed_dict={X_darn: state,
                                                      Y_darn: r_hat, action_true: action})
		print("Epoch= "+str(epoch)+", Episode= " + str(ep) + ", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))
	if(epoch%(num_epochs//10) == 0):
		darn_save_path = saver_darn.save(sess_darn, darn_model_path)#, global_step=epoch)
		print("Model saved in file: %s" % darn_save_path)

	# ind =  random.sample(range(0, x_test.shape[0]), test_batch_size)
	# x_test_batch = x_test[ind,:,:,:]
	# y_test_true_batch = y_test[ind,:]	

	# print("Testing Accuracy "+str(epoch)+": ", sess.run(accuracy, feed_dict={X: x_test_batch,
 #                              Y: y_test_true_batch}))



result = np.zeros(num_test)
count_max = 100
same_image = None
if same:
	image_set,action_set = processGIF('DataSetGenerator/expert_data/'+location+"/"+str(0),100)
	pixels = image_set[0,:,:,:]
	pixels = pixels.squeeze(axis=2)
	for i in range(100):
		for j in range(100):
			if(pixels[j,i] == 64):
				pixels[j, i] = 255

	same_image = Image.fromarray(pixels, 'L')
	#same_image.show()
	#input("Pause")


#Test Network
for t in range(0,num_test):

	data,new_state,gw,failed,done,environment,image = environmentStep(-1,-1,100,100,10,10, image = None, gw = None, environment = None,feed=same_image)
	#image.show()
	#input("Pause")
	#print(t,failed,done)
	count = 0
	frames = []
	f = open("results/"+location+"/"+str(t)+".txt",'w')
	while(not done and not failed):# and count < count_max):
        #preprocess
		full_image = Image.fromarray(data.squeeze(axis=2), 'L')
		frames.append(full_image)
		#full_image.show()
		#input("Pause")
		img = preprocessing(full_image,data_size)
		pixels = list(img.getdata())
		pixels = np.array([pixels[j * data_size:(j + 1) * data_size] for j in range(data_size)])
		pixels = pixels[:,:,np.newaxis]
		
		action = sess_darn.run(darn_presoft,feed_dict={X_darn: [pixels]})
		#print(action)
		action = np.argmax(action[0])
		f.write(str(action))
		f.write('\n')
		#print(action)
		data,new_state,gw,failed,done,environment,image = environmentStep(action,new_state,100,100,10,10,image,gw,environment)
		#print(new_state)
		#print(done,failed)
		#img = Image.fromarray(data.squeeze(axis=2), 'L')
		#img.show()
		if(count == count_max):
			failed = 1;

		if(failed):
			result[t] = 0
			if(count >= count_max):
				print(str(t)+": You took too long")
			else:
				print(str(t)+": You hit a wall!")
		elif(done):
			result[t] = 1
			print(str(t)+": You won!")
		#elif(count >= count_max):
			#result[t] = 0
			#print(str(t)+": Took too long!")

		count+=1
	frames[0].save("results/"+location+"/"+str(t)+".gif",save_all=True, append_images=frames[1:])
	f.flush()
	f.close()
	#print(result.size,t+1)



#print(len(result))
result = np.array(result)
win = np.sum(result)
lose = result.size-win
#print(result.size,lose)
print("After %d tests: %d Passed and %d Failed, Accuracy of: %0.2f" % (num_test,win,lose,win/num_test))




