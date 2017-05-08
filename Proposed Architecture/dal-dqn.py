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

from arch import getData

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression



data_loc = "same_1000.h5"
location = "same"
rew_location = "same"
test_image_location = "DataSetGenerator/test_data/same"
test_array = [0,99,90,9,46,22,66,50,58,49]
same = True
data_size = 83
actual_size = 100
num_train = 1000
num_test = 0
num_reward = 10
test_interval = 500
tests = 2
normalize = 1
pooling_bool = False #use pooling
path = os.path.dirname(os.path.realpath(__file__))
netName = "daqn"
log = netName + "_log"
daqn_model_path = path+'\saved-models\daqn\daqn.ckpt'
darn_model_path = path+'\saved-models\darn\darn.ckpt'


num_epochs = 1000
num_epochs_ran = 10000
batch_size = 50
batch_size_ran = 25
test_batch_size = 50

n_classes = 5 #5 actions
learning_rate = 0.1
gamma = 0.9

restore_epoch = num_epochs-1

replay_buffer = 500
replay_state = np.zeros(shape=[replay_buffer,data_size,data_size,1])
replay_action = np.zeros(shape=[replay_buffer,n_classes])
replay_r =np.zeros(shape=[replay_buffer])
replay_state_prime = np.zeros(shape=[replay_buffer,data_size,data_size,1])

expert_replay_state = np.zeros(shape=[0,data_size,data_size,1])
expert_replay_action = np.zeros(shape=[0,n_classes])
expert_replay_r =np.zeros(shape=[0])
expert_replay_state_prime = np.zeros(shape=[0,data_size,data_size,1])

p = 0.1 #expert replay sampling

x_train, y_train, episode_lengths, episode_start, episode_total, episode_total, x_test, y_test,state,action,state_prime,action_prime = getData(data_loc,location,rew_location,data_size,num_train,num_test,num_reward,normalize)


def test_network(epoch, num_test, same, location, test_image_location,test_array, normalize, data_size, actual_size,debug=True):
	result = np.zeros(num_test)
	count_max = 100
	out_file = open("results/"+location+"/"+str(epoch)+"_Results.txt",'w')
	#Test Network
	for t in range(0,num_test):

		image = Image.open(test_image_location+"/"+str(t)+".png")
		state = test_array[t]		
		pixels = list(image.getdata())
		pixels = np.array([pixels[j * actual_size:(j + 1) * actual_size] for j in range(actual_size)],dtype=np.uint8)
		image = Image.fromarray(pixels, 'L')

		f = open("results/"+location+"/"+str(epoch)+"_"+str(t)+".txt",'w')
		data,new_state,gw,failed,done,environment,image = environmentStep(-1,state,100,100,10,10, image = None, gw = None, environment = None,feed=image)
		#image.show()
		#input("Pause")
		#print(t,failed,done)
		count = 0
		frames = []
		
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
			pixels = np.divide(pixels,normalize)
			
			action = sess.run(daqn_presoft,feed_dict={X: [pixels]})
			f.write(str(action))
			f.write('\n')
			action = np.argmax(action[0])
			f.write(str(action))
			f.write('\n')
			#print(action)
			data,new_state,gw,failed,done,environment,image = environmentStep(action,new_state,100,100,10,10,image,gw,environment)

			if(count == count_max):
				failed = 1;

			if(failed):
				result[t] = 0
				if(count >= count_max):
					if debug:
						print(str(t)+": You took too long")
					out_file.write(str(t)+": You took too long")
					out_file.write('\n')
				else:
					if debug:
						print(str(t)+": You hit a wall!")
					out_file.write(str(t)+": You hit a wall!")
					out_file.write('\n')
				f.write("FAIL")
				f.write('\n')
			elif(done):
				result[t] = 1
				if debug:
					print(str(t)+": You won!")
				out_file.write(str(t)+": You won!")
				out_file.write('\n')
				f.write("PASS")
				f.write('\n')
			count+=1
		frames[0].save("results/"+location+"/"+str(epoch)+"_"+str(t)+".gif",save_all=True, append_images=frames[1:])
		f.flush()
		f.close()
	#print(len(result))
	result = np.array(result)
	win = np.sum(result)
	lose = result.size-win
	#print(result.size,lose)
	print("After %d tests: %d Passed and %d Failed, Accuracy of: %0.2f" % (num_test,win,lose,win/num_test))
	out_file.write("After %d tests: %d Passed and %d Failed, Accuracy of: %0.2f" % (num_test,win,lose,win/num_test))
	out_file.write('\n')
	out_file.flush()
	out_file.close()


expert_replay_state = np.zeros(shape=[x_train.shape[0],data_size,data_size,1])
expert_replay_action = np.zeros(shape=[x_train.shape[0],n_classes])
expert_replay_r =np.zeros(shape=[x_train.shape[0]])
expert_replay_state_prime = np.zeros(shape=[x_train.shape[0],data_size,data_size,1])

#create expert replay data
for i in range(0,x_train.shape[0]):
	#expert_replay_state = np.append(expert_replay_state,[x_train[i,:,:,:]],axis=0)
	#expert_replay_action = np.append(expert_replay_action,[y_train[i,:]],axis=0)
	expert_replay_state[i,:,:,:] = x_train[i,:,:,:]
	expert_replay_action[i,:] = y_train[i,:]
	reward = 0
	if(y_train[i,4] == 1):
		reward = 1
		#expert_replay_state_prime[i,:,:,:] = np.append(expert_replay_state_prime,[x_train[i,:,:,:]],axis=0)
		expert_replay_state_prime[i,:,:,:] = x_train[i,:,:,:]
	else:
		#expert_replay_state_prime = np.append(expert_replay_state_prime,[x_train[i+1,:,:,:]],axis=0)
		expert_replay_state_prime[i,:,:,:] = x_train[i+1,:,:,:]
	#expert_replay_r = np.append(expert_replay_r,[reward],axis=0)
	expert_replay_r[i] = reward
	print(i)
	

f = open("results/daqn_logs/"+location+".txt",'w')

graph_daqn = tf.Graph()
with graph_daqn.as_default():
	#Place Holder
	X = tf.placeholder(shape=[None,data_size,data_size,1], dtype="float32",name='s')
	X = tf.reshape(X, [-1, data_size, data_size, 1])
	#X_prime = tf.placeholder(shape=[None,data_size,data_size,1], dtype="float32",name='s_prime')
	#X_prime = tf.reshape(X, [-1, data_size, data_size, 1])

	Qout = tf.placeholder(shape=[None],dtype="float32",name='Q')
	Q_prime = tf.placeholder(shape=[None],dtype="float32",name='Q_prime')

	R = tf.placeholder(shape=[None],dtype="float32",name='R')

	Y = tf.placeholder(shape=[None,n_classes], dtype="float32", name='a')
	action_true = tf.placeholder(shape=[None,n_classes], dtype="float32", name='action_true')
	net = DAQN(X,Y,pooling_bool)
	daqn = net.outpost
	daqn_presoft = net.outpre
	

	Q = tf.reduce_sum(tf.multiply(daqn_presoft, Y), reduction_indices=1)

	# Define Loss and optimizer
	#cost_0 = tf.multiply(gamma,Q_prime)
	cost_1 = tf.add(R,Q_prime)
	cost_2 = tf.subtract(cost_1,Q)
	cost = tf.square(cost_2)

	cost = tf.reduce_mean(cost)

	optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
	update = optimizer.minimize(cost)
	
	# Evaluate model
	pred = tf.argmax(daqn, 1)
	#pred = tf.Print(pred,[pred],message="prediction is: ")
	true = tf.argmax(Y, 1)
	#true = tf.Print(true,[true],message="truth is: ")

	correct_pred = tf.equal(pred, true) #1 instead of 0?
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	test_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
	#Grab Summaries
	#tf.summary.scalar('loss', cost)
	#tf.summary.scalar('accuracy', accuracy)
	#tf.summary.scalar('test_accuracy', test_accuracy)

	merged = tf.summary.merge_all()	

	init_daqn = tf.global_variables_initializer()
	saver_daqn = tf.train.Saver()

with tf.Session(graph=graph_daqn) as sess:
	sess.run(init_daqn)	
	writer = tf.summary.FileWriter(log,graph=sess.graph)
	
	for epoch in range(num_epochs):
		ind =  np.random.choice(range(0, expert_replay_state.shape[0]),replace=False, size=batch_size)#random.sample(range(0, num), 1)[0]

		s = expert_replay_state[ind,:,:,:]
		a = expert_replay_action[ind,:]

		#s = sess.run(daqn_presoft,feed_dict={X: st, Y: a})

		#q = s[np.arange(s.shape[0]),np.argmax(a,axis=1)]

		s_prime = expert_replay_state_prime[ind,:,:,:]

		s_prime = sess.run(daqn_presoft,feed_dict={X: s_prime, Y: a})

		q_prime = s_prime[np.arange(s_prime.shape[0]),np.argmax(s_prime,axis=1)]

		q_prime = np.multiply(gamma, q_prime)

		r = expert_replay_r[ind]

		sess.run(update,feed_dict={X : s, Y: a, Q_prime : q_prime, R : r})

		# Display loss and accuracy
		loss, acc = sess.run([cost, accuracy], feed_dict={X: s,
                                                      Y: a, action_true: a, Q_prime : q_prime, R : r})
		#writer.add_summary(summary)
		print("Epoch= "+str(epoch)+", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))
		# f.write("Epoch= "+str(epoch)+", Minibatch Loss= " + \
  #             "{:.6f}".format(loss) + ", Training Accuracy= " + \
  #             "{:.5f}".format(acc))
		# f.write('\n')
		if(epoch%(num_epochs//10) == -1):
			daqn_save_path = saver_daqn.save(sess, daqn_model_path)#, global_step=epoch)
			print("Model saved in file: %s" % daqn_save_path)

			ind =  np.random.choice(range(0, x_test.shape[0]),replace=False,size=test_batch_size)
			x_test_batch = x_test[ind,:,:,:]
			y_test_true_batch = y_test[ind,:]	

			val = sess.run([test_accuracy], feed_dict={X: x_test_batch,
		                              Y: y_test_true_batch, action_true: y_test_true_batch})
			#writer.add_summary(summary, epoch)
			print("Testing Accuracy "+str(epoch)+": ", val)
			# f.write("Testing Accuracy "+str(epoch)+": "+ str(val))
			# f.write('\n')
	daqn_save_path = saver_daqn.save(sess, daqn_model_path)#, global_step=epoch)
	print("Model saved in file: %s" % daqn_save_path)

	count_max = 100
	same_image = None
	failed = True
	done = True

	total_step_count = 0

	if same:
		image_set,action_set = processGIF('DataSetGenerator/expert_data/'+location+"/"+str(0),100)
		pixels = image_set[0,:,:,:]
		pixels = pixels.squeeze(axis=2)
		for i in range(actual_size):
			for j in range(actual_size):
				if(pixels[j,i] == 64):
					pixels[j, i] = 255

		same_image = Image.fromarray(pixels, 'L')
	for epoch in range(0,num_epochs_ran):
		
		if (done or failed):# and count < count_max):	
			data,new_state,gw,failed,done,environment,image = environmentStep(-1,-1,100,100,10,10, image = None, gw = None, environment = None,feed=same_image)
			count = 0	
			while(done or failed):
				data,new_state,gw,failed,done,environment,image = environmentStep(-1,-1,100,100,10,10, image = None, gw = None, environment = None,feed=same_image)

		full_image = Image.fromarray(data.squeeze(axis=2), 'L')
		img = preprocessing(full_image,data_size)
		pixels = list(img.getdata())
		pixels = np.array([pixels[j * data_size:(j + 1) * data_size] for j in range(data_size)])
		pixels = pixels[:,:,np.newaxis]
		pixels = np.divide(pixels,normalize)

		replay_state[total_step_count%replay_buffer,:,:,:] = pixels

		action = sess.run(daqn_presoft,feed_dict={X: [pixels]})
		action = np.argmax(action[0:4])

		action_list = np.zeros(shape=[1,5])
		action_list[0,action] = 1

		replay_action[total_step_count%replay_buffer,:] = action_list

		data,new_state,gw,failed,done,environment,image = environmentStep(action,new_state,100,100,10,10,image,gw,environment)

		if(count == count_max):
			failed = 1;

		count += 1
		full_image = Image.fromarray(data.squeeze(axis=2), 'L')
		img = preprocessing(full_image,data_size)
		pixels = list(img.getdata())
		pixels = np.array([pixels[j * data_size:(j + 1) * data_size] for j in range(data_size)])
		pixels = pixels[:,:,np.newaxis]
		pixels = np.divide(pixels,normalize)

		replay_state_prime[total_step_count%replay_buffer,:,:,:] = pixels

		reward = 0

		if failed:
			reward = -1
		elif done:
			reward =1

		replay_r[total_step_count%replay_buffer] = reward

		if(total_step_count > replay_buffer):
			expert_batch = int(p*batch_size_ran)
			agent_batch = batch_size_ran - expert_batch
			ind_expert_data =  np.random.choice(range(0, expert_replay_state.shape[0]),replace=False, size=expert_batch)
			ind_agent_data =  np.random.choice(range(0, replay_state.shape[0]),replace=False, size=agent_batch)

			s = np.concatenate((expert_replay_state[ind_expert_data,:,:,:],replay_state[ind_agent_data,:,:,:]),axis=0)
			a = np.concatenate((expert_replay_action[ind_expert_data,:],replay_action[ind_agent_data,:]),axis=0)

			# s = sess.run(daqn_presoft,feed_dict={X: s, Y: a})

			# q = s[np.arange(s.shape[0]),np.argmax(a,axis=1)]

			s_prime = np.concatenate((expert_replay_state_prime[ind_expert_data,:,:,:],replay_state_prime[ind_agent_data,:,:,:]),axis=0)

			s_prime = sess.run(daqn_presoft,feed_dict={X: s_prime, Y: a})

			q_prime = s_prime[np.arange(s_prime.shape[0]),np.argmax(s_prime,axis=1)]

			q_prime = np.multiply(gamma, q_prime)

			r = np.concatenate((expert_replay_r[ind_expert_data],replay_r[ind_agent_data]),axis=0)

			sess.run(update,feed_dict={X : s, Y: a, Q_prime : q_prime, R : r})

			# Display loss and accuracy
			loss, acc = sess.run([cost, accuracy], feed_dict={X: s,
                                                  Y: a, action_true: a, Q_prime : q_prime, R : r})
			#writer.add_summary(summary)
			print("Epoch= "+str(epoch)+", Minibatch Loss= " + \
	              "{:.6f}".format(loss) + ", Training Accuracy= " + \
	              "{:.5f}".format(acc))

		total_step_count += 1
		
		if(epoch%(num_epochs//10) == -1):
			daqn_save_path = saver_daqn.save(sess, daqn_model_path)#, global_step=epoch)
			print("Model saved in file: %s" % daqn_save_path)

			ind =  np.random.choice(range(0, x_test.shape[0]),replace=False,size=test_batch_size)
			x_test_batch = x_test[ind,:,:,:]
			y_test_true_batch = y_test[ind,:]	

			val = sess.run([test_accuracy], feed_dict={X: x_test_batch,
		                              Y: y_test_true_batch, action_true: y_test_true_batch})
			writer.add_summary(summary, epoch)
			print("Testing Accuracy "+str(epoch)+": ", val)
			# f.write("Testing Accuracy "+str(epoch)+": "+ str(val))
			# f.write('\n')	
		if(epoch%test_interval == 0):
			# Test Network
			test_network(epoch,len(test_array), same, location,test_image_location,test_array, normalize, data_size, actual_size)

		# Test Network
	test_network(epoch,len(test_array), same, location, test_image_location,test_array, normalize, data_size, actual_size)	

# f.flush()
# f.close()
writer.flush()
writer.close()










