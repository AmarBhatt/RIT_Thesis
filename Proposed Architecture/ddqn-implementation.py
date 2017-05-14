import numpy as np
np.set_printoptions(threshold=np.nan)
import tflearn
import tensorflow as tf
from DataSetGenerator.maze_generator import *
from ddqn import *
from test_network import test_network
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

typeTest = "random"


data_loc = typeTest+"_8000_rew_84_skipgoal.h5"#"same_1000_rew_83_skipgoal.h5"
location = typeTest
rew_location = typeTest
test_image_location = "DataSetGenerator/test_data/"+typeTest
test_array = [0,9,99,54,26,35,6,90,21,89] if typeTest == "random" else [0,99,90,9,46,22,66,50,58,49]
same = True if typeTest == "same" else False
skip_goal = -1 #None if you do not want to skip the goal state, -1 if you do (if -1 then possible actions = 4 not 5)

data_size = 84
actual_size = 100
num_train = 8000
num_test = 2000
num_reward = 10
test_interval = 100
tests = 2
normalize = 1
pooling_bool = False #use pooling
path = os.path.dirname(os.path.realpath(__file__))
netName = "dqn"
log = netName + "_log"
dqn_model_path = path+'\saved-models\dqn\dqn.ckpt'
darn_model_path = path+'\saved-models\darn\darn.ckpt'


num_epochs = 10000
num_epochs_ran = 10000
batch_size = 32
batch_size_ran = 32

n_classes = 5 if skip_goal == None else 4
learning_rate = 0.001
gamma = 0.9

REWARD = 1000

p = 0.1 #expert replay sampling

update_freq = 4
tau = 0.001

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



x_train, y_train, episode_lengths, episode_start, episode_total, episode_total, x_test, y_test,state,action,state_prime,action_prime = getData(data_loc,location,rew_location,data_size,num_train,num_test,num_reward,skip_goal=skip_goal,normalize=normalize)


expert_replay_state = np.zeros(shape=[x_train.shape[0],data_size,data_size,1])
expert_replay_action = np.zeros(shape=[x_train.shape[0],n_classes])
expert_replay_r =np.zeros(shape=[x_train.shape[0]])
expert_replay_state_prime = np.zeros(shape=[x_train.shape[0],data_size,data_size,1])

#create expert replay data
episode_start_index = 1
for i in range(0,x_train.shape[0]):
	print(i)
	expert_replay_state[i,:,:,:] = x_train[i,:,:,:]
	expert_replay_action[i,:] = y_train[i,:]
	reward = 0
	if(skip_goal == -1):
		if (i == x_train.shape[0]-1):
			reward = REWARD
			expert_replay_state_prime[i,:,:,:] = x_train[i,:,:,:]
		elif(i == episode_start[episode_start_index]-1):
			reward = REWARD
			expert_replay_state_prime[i,:,:,:] = x_train[i,:,:,:]
			#print(x_train[i,:,:,:].squeeze(axis=2))
			#pix2img(x_train[i,:,:,:].squeeze(axis=2),True)
			#input("pause")
			episode_start_index = min(episode_start_index+1,len(episode_start)-1)
		else:
			expert_replay_state_prime[i,:,:,:] = x_train[i+1,:,:,:]
	else:
		if(y_train[i,4] == 1):
			reward = REWARD
			expert_replay_state_prime[i,:,:,:] = x_train[i,:,:,:]
		else:
			expert_replay_state_prime[i,:,:,:] = x_train[i+1,:,:,:]
	expert_replay_r[i] = reward
	

f = open("results/dqn_logs/"+location+".txt",'w')

graph_dqn = tf.Graph()
with graph_dqn.as_default():
	#Place Holder
	X = tf.placeholder(shape=[None,data_size,data_size,1], dtype="float32",name='s')
	X = tf.reshape(X, [-1, data_size, data_size, 1])
	#X_prime = tf.placeholder(shape=[None,data_size,data_size,1], dtype="float32",name='s_prime')
	#X_prime = tf.reshape(X, [-1, data_size, data_size, 1])
	Y = tf.placeholder(shape=[None,n_classes], dtype="float32", name='a')

	net = DDQN(X,Y,n_classes,learning_rate,pooling_bool=pooling_bool)
	netTarget = DDQN(X,Y,n_classes,learning_rate,pooling_bool=pooling_bool)
    
	# Evaluate model
	#pred = tf.argmax(dqn, 1)
	#pred = tf.Print(pred,[pred],message="prediction is: ")
	true = tf.argmax(Y, 1)
	#true = tf.Print(true,[true],message="truth is: ")

	correct_pred = tf.equal(net.predict, true) #1 instead of 0?
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	test_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
	#Grab Summaries
	#tf.summary.scalar('loss', cost)
	#tf.summary.scalar('accuracy', accuracy)
	#tf.summary.scalar('test_accuracy', test_accuracy)

	merged = tf.summary.merge_all()	

	init_dqn = tf.global_variables_initializer()
	saver_dqn = tf.train.Saver()
	trainables = tf.trainable_variables()

	targetOps = updateTargetGraph(trainables,tau)

with tf.Session(graph=graph_dqn) as sess:
	sess.run(init_dqn)	
	writer = tf.summary.FileWriter(log,graph=sess.graph)
	
	updateTarget(targetOps,sess)
	for epoch in range(num_epochs):
		ind =  np.random.choice(range(0, expert_replay_state.shape[0]),replace=False, size=batch_size)#random.sample(range(0, num), 1)[0]

		s = expert_replay_state[ind,:,:,:]
		a = expert_replay_action[ind,:]

		s_prime = expert_replay_state_prime[ind,:,:,:]

		r = expert_replay_r[ind]



		Q1 = sess.run(net.predict,feed_dict={X:s_prime})
		Q2 = sess.run(netTarget.Qout,feed_dict={X:s_prime})

		#print(Q1.shape)
		#print(Q2.shape)
		#end_multiplier = -(r-1)
		doubleQ = Q2[np.arange(Q2.shape[0]),Q1]#Q2[0:batch_size,Q1]
		targetQ = r + (gamma*doubleQ)#*end_multiplier)

		#print(targetQ.shape)

		sess.run(net.updateModel,feed_dict={X:s,net.targetQ:targetQ,net.actions_onehot:a})

		if(epoch % update_freq == 0):
			updateTarget(targetOps,sess)

		# Display loss and accuracy
		cost, acc = sess.run([net.loss, accuracy], feed_dict={X:s,net.targetQ:targetQ,net.actions_onehot:a, Y: a})
		#writer.add_summary(summary)
		print("Epoch= "+str(epoch)+", Minibatch Loss= " + \
              "{:.6f}".format(cost) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))
		# f.write("Epoch= "+str(epoch)+", Minibatch Loss= " + \
  #             "{:.6f}".format(cost) + ", Training Accuracy= " + \
  #             "{:.5f}".format(acc))
		# f.write('\n')
		if(epoch%(num_epochs//10) == -1):
			dqn_save_path = saver_dqn.save(sess, dqn_model_path)#, global_step=epoch)
			print("Model saved in file: %s" % dqn_save_path)

			ind =  np.random.choice(range(0, x_test.shape[0]),replace=False,size=test_batch_size)
			x_test_batch = x_test[ind,:,:,:]
			y_test_true_batch = y_test[ind,:]	

			val = sess.run([test_accuracy], feed_dict={X: x_test_batch,
		                              Y: y_test_true_batch, action_true: y_test_true_batch})
			#writer.add_summary(summary, epoch)
			print("Testing Accuracy "+str(epoch)+": ", val)
			# f.write("Testing Accuracy "+str(epoch)+": "+ str(val))
			# f.write('\n')
	dqn_save_path = saver_dqn.save(sess, dqn_model_path)#, global_step=epoch)
	print("Model saved in file: %s" % dqn_save_path)

	count_max = 50
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
	
	best_epoch = 0
	best_score = 0
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

		action = sess.run(net.Qout,feed_dict={X: [pixels]})
		action = np.argmax(action[0][0:4])

		action_list = np.zeros(shape=[1,n_classes])
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
			reward = -REWARD
		elif done:
			reward = REWARD

		replay_r[total_step_count%replay_buffer] = reward

		if(total_step_count > replay_buffer):
			expert_batch = int(p*batch_size_ran)
			agent_batch = batch_size_ran - expert_batch
			ind_expert_data =  np.random.choice(range(0, expert_replay_state.shape[0]),replace=False, size=expert_batch)
			ind_agent_data =  np.random.choice(range(0, replay_state.shape[0]),replace=False, size=agent_batch)

			s = np.concatenate((expert_replay_state[ind_expert_data,:,:,:],replay_state[ind_agent_data,:,:,:]),axis=0)
			a = np.concatenate((expert_replay_action[ind_expert_data,:],replay_action[ind_agent_data,:]),axis=0)


			s_prime = np.concatenate((expert_replay_state_prime[ind_expert_data,:,:,:],replay_state_prime[ind_agent_data,:,:,:]),axis=0)


			r = np.concatenate((expert_replay_r[ind_expert_data],replay_r[ind_agent_data]),axis=0)

			Q1 = sess.run(net.predict,feed_dict={X:s_prime})
			Q2 = sess.run(netTarget.Qout,feed_dict={X:s_prime})

			#end_multiplier = -(r-1)
			doubleQ = Q2[np.arange(Q2.shape[0]),Q1]#Q2[range(batch_size),Q1]
			targetQ = r + (gamma*doubleQ)#*end_multiplier)


			sess.run(net.updateModel,feed_dict={X:s,net.targetQ:targetQ,net.actions_onehot:a})

			if(epoch % update_freq == 0):
				updateTarget(targetOps,sess)

			# Display loss and accuracy
			cost, acc = sess.run([net.loss, accuracy], feed_dict={X:s,net.targetQ:targetQ,net.actions_onehot:a, Y: a})
			#writer.add_summary(summary)
			print("Epoch= "+str(epoch)+", Minibatch Loss= " + \
	              "{:.6f}".format(cost) + ", Training Accuracy= " + \
	              "{:.5f}".format(acc))

		total_step_count += 1
		
		if(epoch%(num_epochs//10) == -1):
			dqn_save_path = saver_dqn.save(sess, dqn_model_path)#, global_step=epoch)
			print("Model saved in file: %s" % dqn_save_path)

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
			win,lose,cur_path_total,total_path = test_network(sess,netTarget.Qout, X, epoch,len(test_array), same, location,test_image_location,test_array, normalize, data_size, actual_size)
			if(best_score < cur_path_total):
				best_score = cur_path_total
				best_epoch = epoch

	# Test Network
	win,lose,cur_path_total,total_path = test_network(sess,netTarget.Qout, X, epoch,len(test_array), same, location,test_image_location,test_array, normalize, data_size, actual_size)
	if(best_score < cur_path_total):
		best_score = cur_path_total
		best_epoch = epoch

	print("Best Epoch, Best Score")
	print(best_epoch,best_score)

# f.flush()
# f.close()
writer.flush()
writer.close()










