from deepmind_network import DeepMindNetwork
import numpy as np
import theano
import theano.tensor as T
import os
import os.path
from matplotlib import pyplot as plt
import generalUtils
from mlp import HiddenLayer
import deepmind_static_training as dst
import argparse

class RewardFunction(object):
	NEW_LAYER_SIZE = 30
	def __init__(
		self, network_file, game, history_size, non_linearity,
		batch_size = 32, discount_factor = 0.9, learning_rate = 0.1):

		game_settings = generalUtils.get_game_settings(game)
		num_actions = len(game_settings.possible_actions)
		self.batch_size = batch_size
		self.history_size = history_size
		###############################
		# Construct network structure #
		###############################
		state = T.tensor4('curr_state')
		state_prime = T.tensor4('next_state')
		action = T.ivector('action')
		rng = np.random.RandomState(23455)
		self.r_net = DeepMindNetwork.readFromFile(
			x = state,
			num_actions = num_actions,
			batch_size = batch_size,
			rng = rng,
			history_size = history_size,
			fileName = network_file,
			non_linearity = non_linearity)

		q_net = DeepMindNetwork.readFromFile(
			x = state,
			num_actions = num_actions,
			batch_size = batch_size,
			rng = rng,
			history_size = history_size,
			fileName = network_file,
			non_linearity = non_linearity)
		q_net_prime = DeepMindNetwork.readFromFile(
			x = state_prime,
			num_actions = num_actions,
			batch_size = batch_size,
			rng = rng,
			history_size = history_size,
			fileName = network_file,
			non_linearity = non_linearity)

		#############################
		# Create training procedure #
		#############################

		# TODO: Might need to take [0] on some axis
		# it should return array with size 1 for each
		# element in batch insted of just a number.
		# TODO: "batch_size" instead of "32".
		r_s_a = T.reshape(self.r_net.layer3.before_softmax, (32,))

		q_s_a = q_net.layer3.before_softmax[np.arange(32), action]
		max_q_s_prime_a_prime = T.max(q_net_prime.layer3.before_softmax, axis=1)
		cost = T.sum(T.sqr(r_s_a - (q_s_a - discount_factor * max_q_s_prime_a_prime)))
		#self.debug = theano.function(
		# [state, state_prime, action],
		# [distance],
		# allow_input_downcast = True)

		# This is cost for the whole batch. "distance"
		# is an array with same size as the batch.
		# TODO: L1 instead od L2?
		#cost = distance.norm(L = 2)
		updates = dst.sgd_updates_adagrad(
			cost = cost,
			params = self.r_net.params,
			learning_rate = learning_rate,
			consider_constant=[q_s_a, max_q_s_prime_a_prime])
		self.train_model = theano.function(
			[state, state_prime, action],
			cost,
			updates = updates,
			allow_input_downcast = True)
	
	@staticmethod
	def load_transitions_from_file(fileName, history_size):
		state = np.load(fileName)['curr_state']
		state_prime = state[1:]
		state = state[:state.shape[0] - 1]
		action = np.load(fileName)['action']
		action = action[:action.shape[0] - 1]


		# TODO: Shuffle. I should share load code
		# with deepmind_static_training2.py
		return [dst.create_multiframe_states(state, history_size), dst.create_multiframe_states(state_prime, history_size), action]

	def train(self, transition_data_folder):

		num_transition_files = generalUtils.get_num_files(transition_data_folder, '.npz')

		for k in range(num_transition_files):
			random_index = k #np.random.randint(low = 0, high = num_transition_files)
			[state, state_prime, action] = RewardFunction.load_transitions_from_file(
				transition_data_folder + str(random_index) + '.npz',
				self.history_size)
			print 'using episode ' + str(random_index)
			for i in range(200):
				print 'batch: ' + str(i)
				indices = np.random.randint(low = 0, high = state.shape[0], size = self.batch_size)
				# print self.debug(state[indices], state_prime[indices], action[indices])
				self.train_model(state[indices], state_prime[indices], action[indices])
			self.r_net.saveToFile(transition_data_folder + 'r_net_' + str(k))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Generates gameplay from a given network')
	parser.add_argument('-g','--game', help='Which game', required=True)
	parser.add_argument('-s','--history_size', help='Number of frames in input', required=True, type = int)
	parser.add_argument('-n','--network', help='Network location', required=True)
	parser.add_argument('-d','--database', help='Human gameplay database', required=True)
	parser.add_argument('-l','--non_linearity', help='tanh or ReLU', required=True)

	args = vars(parser.parse_args())

	reward_function = RewardFunction(args['network'], args['game'], args['history_size'], args['non_linearity'])
	reward_function.train(args['database'])