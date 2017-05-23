import numpy as np
np.set_printoptions(threshold=np.nan)
import tflearn
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMCell
from DataSetGenerator.maze_generator import *
from drqn import *
from test_network import test_network_drqn
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

typeTest = "same"


data_loc = typeTest+"_1000_rew_84_skipgoal.h5"
location = typeTest
rew_location = typeTest
test_image_location = "DataSetGenerator/test_data/"+typeTest
test_array = [0,9,99,54,26,35,6,90,21,89] if typeTest == "random" else [0,99,90,9,46,22,66,50,58,49]
same = True if typeTest == "same" else False
skip_goal = -1 #None if you do not want to skip the goal state, -1 if you do (if -1 then possible actions = 4 not 5)

data_size = 84
actual_size = 100
num_train = 8000
num_test = 0
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


num_epochs = 2000
num_epochs_ran = 100000
max_batch_size = 8
max_batch_size_ran = 8
max_trace_length = 4

n_classes = 5 if skip_goal == None else 4
learning_rate = 0.0001
gamma = 0.9

REWARD = 1000

p = 0.125 #expert replay sampling
decay_rate = 0.25
decay_frequency = 2500000000

update_freq = 5
tau = 0.001
h_size = 512

restore_epoch = num_epochs-1

replay_buffer = 500
# replay_state = np.zeros(shape=[replay_buffer,data_size,data_size,1])
# replay_action = np.zeros(shape=[replay_buffer,n_classes])
# replay_r =np.zeros(shape=[replay_buffer])
# replay_state_prime = np.zeros(shape=[replay_buffer,data_size,data_size,1])
agent_episode_length = np.zeros(shape=[replay_buffer])
agent_episode_start = np.zeros(shape=[replay_buffer])

replay_state = np.zeros(shape=[replay_buffer,100,data_size,data_size,1])
replay_action = np.zeros(shape=[replay_buffer,100,n_classes])
replay_r =np.zeros(shape=[replay_buffer,100])
replay_state_prime = np.zeros(shape=[replay_buffer,100,data_size,data_size,1])
replay_all = [replay_state,replay_action,replay_state_prime,replay_r] #np.empty(shape=[replay_buffer,4]) #state, action, state', r


x_train, y_train, episode_lengths, episode_start, episode_total, episode_total, x_test, y_test,state,action,state_prime,action_prime = getData(data_loc,location,rew_location,data_size,num_train,num_test,num_reward,skip_goal=skip_goal,normalize=normalize)

episode_lengths = np.array(episode_lengths)
episode_start = np.array(episode_start)

possible_lengths = np.unique(episode_lengths)

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

	cell = tf.contrib.rnn.LSTMCell(num_units=h_size,state_is_tuple=True)
	cellT = tf.contrib.rnn.LSTMCell(num_units=h_size,state_is_tuple=True)
	net = DRQN(X,Y,n_classes,cell,learning_rate,pooling_bool=pooling_bool,h_size=h_size,myScope="mainQN")
	netTarget = DRQN(X,Y,n_classes,cellT,learning_rate,pooling_bool=pooling_bool,h_size=h_size,myScope="targetQN")
    
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

		# Get trace length
		ep_length = np.random.choice(possible_lengths, replace=False,size=1)[0]

		while(ep_length < 2):
			ep_length = np.random.choice(possible_lengths, replace=False,size=1)[0]

		if ep_length % 2 != 0:
			ep_length -= 1


		# Find all indices of that trace length or higher
		indLengths = np.where(episode_lengths >= ep_length)[0]
		batch_size = min(len(indLengths),max_batch_size)
		trace_length = min(ep_length,max_trace_length)


		#print(len(indLengths),batch_size,ep_length,trace_length)

		#Reset the recurrent layer's hidden state
		state_train = (np.zeros([batch_size,h_size]),np.zeros([batch_size,h_size]))

		#Get all the starting positions
		start_positions = episode_start[indLengths]
		np.random.shuffle(start_positions)

		# Get all trace data
		s = np.empty(shape=[trace_length*batch_size,data_size,data_size,1])
		a = np.empty(shape=[trace_length*batch_size,n_classes])
		s_prime = np.empty(shape=[trace_length*batch_size,data_size,data_size,1])
		r = np.empty(shape=[trace_length*batch_size])

		for b in range(0,batch_size):
			#print(b)
			point = np.random.randint(start_positions[b],start_positions[b]+ep_length+1-trace_length)

			indStart = b*trace_length
			indEnd = b*trace_length+trace_length

			s[indStart:indEnd,:,:,:] = expert_replay_state[point:point+trace_length,:,:,:]
			a[indStart:indEnd,:] = expert_replay_action[point:point+trace_length,:]
			s_prime[indStart:indEnd,:,:,:] = expert_replay_state_prime[point:point+trace_length,:,:,:]
			r[indStart:indEnd] = expert_replay_r[point:point+trace_length]

		# print(indLengths)
		# print(s.shape)
		# for m in range(s.shape[0]):
		# 	pix2img(s[m,:,:,:].squeeze(axis=2),True)
		# 	print(a[m,:])
		# 	input("pause")
		# print(s_prime.shape)
		# for m in range(s_prime.shape[0]):
		# 	pix2img(s_prime[m,:,:,:].squeeze(axis=2),True)
		# 	print(r[m])
		# 	input("pause")

		Q1 = sess.run(net.predict,feed_dict={X:s_prime,net.trainLength:trace_length,net.state_in:state_train,net.batch_size:batch_size})
		Q2 = sess.run(netTarget.Qout,feed_dict={X:s_prime,netTarget.trainLength:trace_length,netTarget.state_in:state_train,netTarget.batch_size:batch_size})

		#print(Q1)
		#print(Q2)

		#print(Q1.shape)
		#print(Q2.shape)
		#end_multiplier = -(r-1)
		doubleQ = Q2[np.arange(Q2.shape[0]),Q1]#Q2[0:batch_size,Q1]
		targetQ = r + (gamma*doubleQ)#*end_multiplier)


		#print(doubleQ)
		#print(targetQ)
		#input("pause")

		sess.run(net.updateModel,feed_dict={X:s,net.targetQ:targetQ,net.actions_onehot:a,net.trainLength:trace_length,net.state_in:state_train,net.batch_size:batch_size})

		if(epoch % update_freq == 0):
			updateTarget(targetOps,sess)

		# Display loss and accuracy
		cost, acc = sess.run([net.loss, accuracy], feed_dict={X:s,net.targetQ:targetQ,net.actions_onehot:a, Y: a,net.trainLength:trace_length,net.state_in:state_train,net.batch_size:batch_size})
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
	increment = 0
	buffer_full = False

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

			#agent_episode_start[total_step_count%replay_buffer] = total_step_count%replay_buffer
			agent_episode_start[increment] = increment
			#length_holder = total_step_count%replay_buffer
			length_holder = increment
			state_holder = np.empty(shape=[100,data_size,data_size,1])
			action_holder = np.empty(shape=[100,n_classes])
			state_prime_holder = np.empty(shape=[100,data_size,data_size,1])
			reward_holder = np.empty(shape=[100])
			state_reset = (np.zeros([1,h_size]),np.zeros([1,h_size]))

		full_image = Image.fromarray(data.squeeze(axis=2), 'L')
		img = preprocessing(full_image,data_size)
		pixels = list(img.getdata())
		pixels = np.array([pixels[j * data_size:(j + 1) * data_size] for j in range(data_size)])
		pixels = pixels[:,:,np.newaxis]
		pixels = np.divide(pixels,normalize)

		#replay_state[total_step_count%replay_buffer,:,:,:] = pixels

		state_holder[count,:,:,:] = pixels

		#action = sess.run(net.Qout,feed_dict={X: [pixels]})
		action = sess.run(net.Qout,feed_dict={X: [pixels],net.trainLength:1,net.state_in:state_reset,net.batch_size:1})
		action = np.argmax(action[0])

		action_list = np.zeros(shape=[1,n_classes])
		action_list[0,action] = 1

		#replay_action[total_step_count%replay_buffer,:] = action_list
		action_holder[count,:] = action_list


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

		#replay_state_prime[total_step_count%replay_buffer,:,:,:] = pixels
		state_prime_holder[count-1,:,:,:] = pixels


		reward = 0

		if failed:
			reward = -REWARD
			state_prime_holder[count-1,:,:,:] = state_holder[count-1,:,:,:]
			#agent_episode_length[length_holder] = count
		elif done:
			reward = REWARD
			state_prime_holder[count-1,:,:,:] = state_holder[count-1,:,:,:]
			#agent_episode_length[length_holder] = count

		reward_holder[count-1] = reward

		if failed or done:
			#Update episodes
			# space_left_in_buffer = replay_buffer - length_holder
			episode_length = count
			# if(episode_length > space_left_in_buffer):
				# episode_length = space_left_in_buffer
			#agent_episode_length[length_holder] = episode_length
			agent_episode_length[increment] = episode_length

			# replay_state[length_holder:length_holder+episode_length,:,:,:] = state_holder[0:episode_length,:,:,:]
			# replay_action[length_holder:length_holder+episode_length,:] = action_holder[0:episode_length,:]
			# replay_state_prime[length_holder:length_holder+episode_length,:,:,:] = state_prime_holder[0:episode_length,:,:,:]
			# replay_r[length_holder:length_holder+episode_length] = reward_holder[0:episode_length]

			state_holder.resize(episode_length,data_size,data_size,1)
			action_holder.resize(episode_length,n_classes)
			state_prime_holder.resize(episode_length,data_size,data_size,1)
			reward_holder.resize(episode_length)

			# print(episode_length)
			# print(state_holder.shape)
			# for m in range(state_holder.shape[0]):
			# 	pix2img(state_holder[m,:,:,:].squeeze(axis=2),True)
			# 	print(action_holder[m,:])
			# 	input("pause")
			# print(state_prime_holder.shape)
			# for m in range(state_prime_holder.shape[0]):
			# 	pix2img(state_prime_holder[m,:,:,:].squeeze(axis=2),True)
			# 	print(reward_holder[m])
			# 	input("pause")

			replay_all[0][increment,0:episode_length,:,:,:] = state_holder
			replay_all[1][increment,0:episode_length,:] = action_holder
			replay_all[2][increment,0:episode_length,:,:,:] = state_prime_holder
			replay_all[3][increment,0:episode_length] = reward_holder

			increment+=1
			if(increment == replay_buffer):
				increment = 0
				buffer_full = True
			#print(increment)

		#replay_r[total_step_count%replay_buffer] = reward
		


		#if(total_step_count > replay_buffer):
		if(buffer_full):

			agent_possible_lengths = np.unique(agent_episode_length)

			# Get trace length
			ep_length = np.random.choice(agent_possible_lengths, replace=False,size=1)[0]

			while(ep_length < 2):
				ep_length = np.random.choice(agent_possible_lengths, replace=False,size=1)[0]

			if ep_length % 2 != 0:
				ep_length -= 1


			trace_length = int(min(ep_length,max_trace_length))

			# Find all indices of that trace length or higher
			indLengths = np.where(agent_episode_length >= trace_length)[0]
			batch_size = min(len(indLengths),max_batch_size_ran)

			
			expert_batch = int(p*batch_size)
			agent_batch = batch_size - expert_batch

			


			#print(len(indLengths),batch_size,ep_length,trace_length,expert_batch,agent_batch)

			#Reset the recurrent layer's hidden state
			state_train = (np.zeros([batch_size,h_size]),np.zeros([batch_size,h_size]))

			#Get all the starting positions
			start_positions = episode_start[indLengths]
			#agent_start_positions = agent_episode_start[indLengths]

			agent_start_positions = indLengths.copy()

			np.random.shuffle(start_positions)
			np.random.shuffle(agent_start_positions)

			# Get all trace data
			s = np.empty(shape=[int(trace_length*batch_size),data_size,data_size,1])
			a = np.empty(shape=[int(trace_length*batch_size),n_classes])
			s_prime = np.empty(shape=[int(trace_length*batch_size),data_size,data_size,1])
			r = np.empty(shape=[int(trace_length*batch_size)])

			for b in range(0,expert_batch):
				point = np.random.randint(start_positions[b],start_positions[b]+ep_length+1-trace_length)
				indStart = b*trace_length
				indEnd = b*trace_length+trace_length

				s[indStart:indEnd,:,:,:] = expert_replay_state[point:point+trace_length,:,:,:]
				a[indStart:indEnd,:] = expert_replay_action[point:point+trace_length,:]
				s_prime[indStart:indEnd,:,:,:] = expert_replay_state_prime[point:point+trace_length,:,:,:]
				r[indStart:indEnd] = expert_replay_r[point:point+trace_length]

			for c in range(expert_batch,batch_size):
				#print(c)
				#point = np.random.randint(agent_start_positions[c-expert_batch],agent_start_positions[c-expert_batch]+ep_length+1-trace_length)
				total_length = agent_episode_length[agent_start_positions[c-expert_batch]]
				point = np.random.randint(0,ep_length+1-trace_length)
				indStart = c*trace_length
				indEnd = c*trace_length+trace_length
				# s[indStart:indEnd,:,:,:] = replay_state[point:point+trace_length,:,:,:]
				# a[indStart:indEnd,:] = replay_action[point:point+trace_length,:]
				# s_prime[indStart:indEnd,:,:,:] = replay_state_prime[point:point+trace_length,:,:,:]
				# r[indStart:indEnd] = replay_r[point:point+trace_length]

				tmp_state = replay_all[0][agent_start_positions[c-expert_batch],point:point+trace_length,:,:,:]
				tmp_action = replay_all[1][agent_start_positions[c-expert_batch],point:point+trace_length,:]
				tmp_state_prime = replay_all[2][agent_start_positions[c-expert_batch],point:point+trace_length,:,:,:]
				tmp_reward = replay_all[3][agent_start_positions[c-expert_batch],point:point+trace_length]

				s[indStart:indEnd,:,:,:] = tmp_state
				a[indStart:indEnd,:] = tmp_action
				s_prime[indStart:indEnd,:,:,:] = tmp_state_prime
				r[indStart:indEnd] = tmp_reward
			#print(agent_episode_start)
			#print(agent_episode_length)
			#print(agent_possible_lengths)
			# print(indLengths)
			# print(agent_start_positions)
			# print(s.shape)
			# for m in range(s.shape[0]):
			# 	pix2img(s[m,:,:,:].squeeze(axis=2),True)
			# 	print(a[m,:])
			# 	input("pause")
			# print(s_prime.shape)
			# for m in range(s_prime.shape[0]):
			# 	pix2img(s_prime[m,:,:,:].squeeze(axis=2),True)
			# 	print(r[m])
			# 	input("pause")

			Q1 = sess.run(net.predict,feed_dict={X:s_prime,net.trainLength:trace_length,net.state_in:state_train,net.batch_size:batch_size})
			Q2 = sess.run(netTarget.Qout,feed_dict={X:s_prime,netTarget.trainLength:trace_length,netTarget.state_in:state_train,netTarget.batch_size:batch_size})

			#print(Q1.shape)
			#print(Q2.shape)
			#end_multiplier = -(r-1)
			doubleQ = Q2[np.arange(Q2.shape[0]),Q1]#Q2[0:batch_size,Q1]
			targetQ = r + (gamma*doubleQ)#*end_multiplier)

			#print(targetQ.shape)

			sess.run(net.updateModel,feed_dict={X:s,net.targetQ:targetQ,net.actions_onehot:a,net.trainLength:trace_length,net.state_in:state_train,net.batch_size:batch_size})


			if(epoch % update_freq == 0):
				updateTarget(targetOps,sess)
			if(epoch % decay_frequency == 0):
				print("Old p: ",p)
				p = max(p-decay_rate,0)
				print("New p: ",p)

			# Display loss and accuracy
			cost, acc = sess.run([net.loss, accuracy], feed_dict={X:s,net.targetQ:targetQ,net.actions_onehot:a, Y: a,net.trainLength:trace_length,net.state_in:state_train,net.batch_size:batch_size})
			#writer.add_summary(summary)
			print("Epoch= "+str(epoch)+", Minibatch Loss= " + \
	              "{:.6f}".format(cost) + ", Training Accuracy= " + \
	              "{:.5f}".format(acc) + ", Best Epoch= " + str(best_epoch) + ", Best Score= " + str(best_score))

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
			win,lose,cur_path_total,total_path = test_network_drqn(sess,netTarget, X, epoch,len(test_array), same, location,test_image_location,test_array, normalize, data_size, actual_size, h_size)
			if(best_score < cur_path_total):
				best_score = cur_path_total
				best_epoch = epoch

	# Test Network
	win,lose,cur_path_total,total_path = test_network_drqn(sess,netTarget, X, epoch,len(test_array), same, location,test_image_location,test_array, normalize, data_size, actual_size, h_size)
	if(best_score < cur_path_total):
		best_score = cur_path_total
		best_epoch = epoch

	print("Best Epoch, Best Score")
	print(best_epoch,best_score)

# f.flush()
# f.close()
writer.flush()
writer.close()










