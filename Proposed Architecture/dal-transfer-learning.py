import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
from DataSetGenerator.maze_generator import *
from daqn import DAQN, sse
from dqn import DQN
from test_network import test_network
import sys as sys
from PIL import Image
import os.path
import h5py
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

from arch import getData

typeTest = "random"

data_loc = typeTest+"_1000_rew_83_skipgoal.h5"#"same_1000_rew_83_skipgoal.h5"
location = typeTest
rew_location = typeTest
test_image_location = "DataSetGenerator/test_data/"+typeTest
test_array = [0,9,99,54,26,35,6,90,21,89] if typeTest == "random" else [0,99,90,9,46,22,66,50,58,49]
same = True if typeTest == "same" else False
skip_goal = -1 #None if you do not want to skip the goal state, -1 if you do (if -1 then possible actions = 4 not 5)
data_size = 83
actual_size = 100
num_train = 8000
num_test = 2000
num_reward = 10000
test_interval = 100
tests = 2
normalize = 1
pooling_bool = False #use pooling
path = os.path.dirname(os.path.realpath(__file__))
netName = "daqn"
log = netName + "_log"
daqn_model_path = path+'\saved-models\daqn\daqn.ckpt'
darn_model_path = path+'\saved-models\darn\darn.ckpt'

num_epochs = 10000
num_epochs_darn = 10000
batch_size = 50
batch_size_darn = 50
test_batch_size = 50

n_classes = 5 if skip_goal == None else 4


learning_rate = 0.01
gamma = 0.9

restore_epoch = num_epochs-1

x_train, y_train, episode_lengths, episode_start, episode_total, episode_total, x_test, y_test,state,action,state_prime,action_prime = getData(data_loc,location,rew_location,data_size,num_train,num_test,num_reward,skip_goal=skip_goal,normalize=normalize)


f = open("results/daqn_logs/"+location+".txt",'w')

graph_daqn = tf.Graph()
with graph_daqn.as_default():
	#Place Holder
	X = tf.placeholder(shape=[None,data_size,data_size,1], dtype="float32",name='s')
	X = tf.reshape(X, [-1, data_size, data_size, 1])

	Y = tf.placeholder(shape=[None,n_classes], dtype="float32", name='a')
	action_true = tf.placeholder(shape=[None,n_classes], dtype="float32", name='action_true')
	net = DAQN(X,Y,n_classes,pooling_bool)
	daqn = net.outpost
	daqn_presoft = net.outpre
	
	# Define Loss and optimizer
	#cost = sse(daqn,Y)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=daqn_presoft,labels=Y)
	cost = tf.reduce_mean(cross_entropy)

	optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate) #tf.train.AdamOptimizer(learning_rate=learning_rate)#
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
	tf.summary.scalar('loss', cost)
	tf.summary.scalar('accuracy', accuracy)
	tf.summary.scalar('test_accuracy', test_accuracy)

	merged = tf.summary.merge_all()


	#X_darn = tf.placeholder(shape=[None,data_size,data_size,1], dtype="float32",name='X_darn')
	#X_darn = tf.reshape(X_darn, [-1, data_size, data_size, 1])
	#Y_darn = tf.placeholder(shape=[None,5], dtype="float32", name='Y_darn')
	#cost_darn = tf.norm(tf.subtract(darn_presoft,Y_darn), ord='euclidean')

	#daqn_presoft_print = tf.Print(daqn_presoft,[daqn_presoft],message="Predicted Values: ")
	#Y_print = tf.Print(Y,[Y],message = "Actual Values: ")

	prediction = tf.multiply(daqn_presoft,action_true)
	# prediction = tf.Print(prediction,[prediction],message="Prediction is: ",summarize=100)
	truth = tf.multiply(Y,action_true)
	# truth = tf.Print(truth,[truth],message="Truth is: ",summarize=100)
	cost_darn_0 = tf.square(tf.subtract(prediction,truth))
	# cost_darn_0 = tf.Print(cost_darn_0,[cost_darn_0],message="cost 0 is: ",summarize=100)
	cost_darn_1 = tf.reduce_max(cost_darn_0,axis=1)
	# cost_darn_1 = tf.Print(cost_darn_1,[cost_darn_1],message="cost 1 is: ",summarize=100)
	cost_darn = tf.reduce_mean(cost_darn_1)
	# cost_darn = tf.Print(cost_darn,[cost_darn],message="cost* is: ",summarize=100)

	optimizer_darn = tf.train.AdagradOptimizer(learning_rate=learning_rate) #tf.train.AdamOptimizer(learning_rate=learning_rate)#
	update_darn = optimizer_darn.minimize(cost_darn)
	
	# Evaluate model
	pred_darn = tf.multiply(daqn_presoft,action_true)
	#pred_darn = tf.Print(pred_darn,[pred_darn],message="prediction is: ")
	true_darn = Y
	#true_darn = tf.Print(true_darn,[true_darn],message="truth is: ")

	correct_pred_darn = tf.subtract(pred_darn, true_darn) #1 instead of 0?
	# correct_pred_darn = tf.Print(correct_pred_darn,[correct_pred_darn],message="Delta: ", summarize=100)
	accuracy_darn_0 = tf.abs(correct_pred_darn)
	# accuracy_darn_0 = tf.Print(accuracy_darn_0,[accuracy_darn_0],message="Abs: ", summarize=100)
	accuracy_darn_1 = tf.reduce_max(accuracy_darn_0,axis=1)
	# accuracy_darn_1 = tf.Print(accuracy_darn_1,[accuracy_darn_1],message="Max: ", summarize=100)
	accuracy_darn = tf.cast(tf.reduce_mean(accuracy_darn_1), tf.float32)
	# accuracy_darn = tf.Print(accuracy_darn,[accuracy_darn],message="Mean: ", summarize=100)
	

	init_daqn = tf.global_variables_initializer()
	saver_daqn = tf.train.Saver()

with tf.Session(graph=graph_daqn) as sess:
	sess.run(init_daqn)	
	writer = tf.summary.FileWriter(log,graph=sess.graph)
	
	for epoch in range(num_epochs):
		ind =  np.random.choice(range(0, x_train.shape[0]),replace=False, size=batch_size)#random.sample(range(0, num), 1)[0]
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
		summary,loss, acc = sess.run([merged,cost, accuracy], feed_dict={X: x_data,
                                                      Y: y_data, action_true: y_data})
		writer.add_summary(summary)
		print("Epoch= "+str(epoch)+", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))
		f.write("Epoch= "+str(epoch)+", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))
		f.write('\n')
		if(epoch%(num_epochs//10) == 0):
			daqn_save_path = saver_daqn.save(sess, daqn_model_path)#, global_step=epoch)
			print("Model saved in file: %s" % daqn_save_path)

			ind =  np.random.choice(range(0, x_test.shape[0]),replace=False,size=test_batch_size)
			x_test_batch = x_test[ind,:,:,:]
			y_test_true_batch = y_test[ind,:]	

			summary,val = sess.run([merged,test_accuracy], feed_dict={X: x_test_batch,
		                              Y: y_test_true_batch, action_true: y_test_true_batch})
			writer.add_summary(summary, epoch)
			print("Testing Accuracy "+str(epoch)+": ", val)
			f.write("Testing Accuracy "+str(epoch)+": "+ str(val))
			f.write('\n')
	daqn_save_path = saver_daqn.save(sess, daqn_model_path)#, global_step=epoch)
	print("Model saved in file: %s" % daqn_save_path)

	ind =  np.random.choice(range(0, x_test.shape[0]),replace=False,size=test_batch_size)
	x_test_batch = x_test[ind,:,:,:]
	y_test_true_batch = y_test[ind,:]	

	summary,val = sess.run([merged,test_accuracy], feed_dict={X: x_test_batch,
                              Y: y_test_true_batch, action_true: y_test_true_batch})
	writer.add_summary(summary, epoch)
	print("Testing Accuracy "+str(epoch)+": ", val)
	f.write("Testing Accuracy "+str(epoch)+": "+ str(val))
	f.write('\n')
f.flush()
f.close()
writer.flush()
writer.close()

f = open("results/darn_logs/"+location+".txt",'w')

sess_daqn = tf.Session(graph=graph_daqn)
sess_daqn.run(init_daqn)
saver_daqn.restore(sess_daqn,daqn_model_path)

sess_darn = tf.Session(graph=graph_daqn)
sess_darn.run(init_daqn)
saver_daqn.restore(sess_darn,daqn_model_path)

best_epoch = 0
best_score = 0

writer_darn = tf.summary.FileWriter('DARN_log',graph=sess.graph)
for epoch in range(num_epochs_darn):
	ind = np.random.choice(range(0,state.shape[0]),replace=False,size=batch_size_darn)
	st = state[ind,:,:,:]
	a = action[ind,:]
	st_p = state_prime[ind,:,:,:]
	a_p = action_prime[ind,:]
	
	Q = sess_daqn.run(daqn_presoft,feed_dict={X: st,Y: a})
	# print(Q,Q.shape, a,np.transpose(np.argmax(a,axis=1)))
	# input("pause")
	Q = Q[np.arange(Q.shape[0]),np.argmax(a,axis=1)]
	# print(Q)
	# input("pause")
	#print(Q)
	Q_p = sess_daqn.run(daqn_presoft,feed_dict={X: st_p,Y: a})
	# print(Q_p)
	# input("pause")
	Q_p = np.amax(Q_p,axis=1)
	# print(Q_p)
	# input("pause")

	# print(Q-(gamma*Q_p))
	# input("pause")
	#print(Q_p)
	r_hat = np.zeros(a.shape)#a
	
	r_hat[np.arange(r_hat.shape[0]),np.argmax(a,axis=1)] = Q-(gamma*Q_p)
	# print("Expected value: ",r_hat)
	# print(a,np.argmax(a,axis=1))
	# input("pause")
	#r_hat = [[r_hat]]

	sess_darn.run(update_darn,feed_dict={X: st, Y: r_hat, action_true: a})
	# input("pause")
	# Display loss and accuracy
	summary,loss, acc = sess_darn.run([merged,cost_darn, accuracy_darn], feed_dict={X: st,
                                                  Y: r_hat, action_true: a})
	writer_darn.add_summary(summary)
	print("Epoch= "+str(epoch)+", Minibatch Loss= " + \
          "{:.6f}".format(loss) + ", Training Accuracy= " + \
          "{:.5f}".format(acc))
	f.write("Epoch= "+str(epoch)+", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
	f.write('\n')
	if(epoch%(num_epochs_darn//10) == 0):
		darn_save_path = saver_daqn.save(sess_darn, darn_model_path)#, global_step=epoch)
		print("Model saved in file: %s" % darn_save_path)
	
	if(epoch%test_interval == 0):
		# Test Network
		win,lose,cur_path_total,total_path = test_network(sess_darn,daqn_presoft, X, epoch,len(test_array), same, location,test_image_location,test_array, normalize, data_size, actual_size)
		if(best_score < cur_path_total):
			best_score = cur_path_total
			best_epoch = epoch

# Test Network
win,lose,cur_path_total,total_path = test_network(sess_darn,daqn_presoft, X, epoch,len(test_array), same, location,test_image_location,test_array, normalize, data_size, actual_size)
if(best_score < cur_path_total):
	best_score = cur_path_total
	best_epoch = epoch

print("Best Epoch, Best Score")
print(best_epoch,best_score)

f.flush()
f.close()
writer_darn.flush()
writer_darn.close()









