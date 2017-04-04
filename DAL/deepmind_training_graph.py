import theano
import theano.tensor as T
import numpy as np
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer
from theano import shared
from logistic_sgd import LogisticRegression
from deepmind_network import DeepMindNetwork
import deepmind_static_training as dst

class DoubleNetwork:
	def __init__(self, num_actions, gama, batch_size, rng, history_size, learning_rate = 0.1):
		self.batch_size = batch_size
		
		curr_state = T.tensor4('curr_state')
		next_state = T.tensor4('next_state')
		reward = T.lvector('reward')
		action = T.lvector('action')
		terminal = T.lvector('terminal')
		
		small_curr_state = T.tensor4('small_curr_state')
		
		self.net1 = DeepMindNetwork(
			x = curr_state, num_actions = num_actions,
			batch_size = batch_size, rng = rng,
			history_size = history_size,
			W0 = None, b0 = None,
			W1 = None, b1 = None,
			W2 = None, b2 = None,
			W3 = None, b3 = None)
		net2 = DeepMindNetwork(
			x = next_state, num_actions = num_actions,
			batch_size = batch_size, rng = rng,
			history_size = history_size,
			W0 = self.net1.layer0.W, b0 = self.net1.layer0.b,
			W1 = self.net1.layer1.W, b1 = self.net1.layer1.b,
			W2 = self.net1.layer2.W, b2 = self.net1.layer2.b,
			W3 = self.net1.layer3.W, b3 = self.net1.layer3.b)
		small_net = DeepMindNetwork(
			x = small_curr_state, num_actions = num_actions,
			batch_size = 1, rng = rng,
			history_size = history_size,
			W0 = self.net1.layer0.W, b0 = self.net1.layer0.b,
			W1 = self.net1.layer1.W, b1 = self.net1.layer1.b,
			W2 = self.net1.layer2.W, b2 = self.net1.layer2.b,
			W3 = self.net1.layer3.W, b3 = self.net1.layer3.b)
		y = T.switch(T.eq(terminal, 1), reward, reward + gama * T.max(net2.layer3.output, axis = 1))
		
		y_pred = self.net1.layer3.output[np.arange(batch_size), action]
		cost = T.sum(T.sqr(y - y_pred))
		
		q = self.net1.layer3.output[np.arange(batch_size), action]
		q_prime = T.max(net2.layer3.output, axis = 1)
		l = y - y_pred
		
		self.params = self.net1.layer3.params + self.net1.layer2.params + self.net1.layer1.params + self.net1.layer0.params
		
		updates = dst.sgd_updates_adagrad(cost = cost, params = self.params, learning_rate = learning_rate, consider_constant = [y])
		
		self.minibatch_update = theano.function(
			[curr_state, next_state, reward, action, terminal],
			cost,
			updates = updates,
			allow_input_downcast = True)
		self.best_action_for_state = theano.function([small_curr_state], T.argmax(small_net.layer3.output, axis = 1), allow_input_downcast=True)
		
		self.get_debug_info = theano.function(
			[curr_state, next_state, reward, action, terminal],
			[q, q_prime, reward, l],
			allow_input_downcast = True)