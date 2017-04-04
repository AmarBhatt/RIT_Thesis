import cPickle
import gzip
import os
import sys
import time
import numpy
import numpy as np
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
import os.path
from PIL import Image
from matplotlib import pyplot as plt
from scipy.misc import imresize
from matplotlib import colors as clrf
from collections import OrderedDict
import theano
import theano.tensor as T
import generalUtils
from deepmind_network import DeepMindNetwork
from os import mkdir
from os.path import exists
import argparse

def loadEpisode(database, episodeIndex, history_size):
	episode_path = database + str(episodeIndex) + '.npz'
	return load_episode_from_path(episode_path, history_size)

def create_multiframe_states(states, history_size):
	#print states.shape
	multiframe_states = np.zeros((states.shape[0], history_size, states.shape[1], states.shape[2]))
	#print multiframe_states.shape
	for i in range(history_size):
		multiframe_states[(history_size - i - 1):, i, :, :] = states[:(states.shape[0] - history_size + i + 1)]
	return multiframe_states

def load_episode_from_path(episode_path, history_size):
	episode = np.load(episode_path)['x']
	# To handle 4 times faster frame rate when recording expert play
	episode = episode[::4]
	background = np.median(episode, axis=0)
	#episode -= background
	episode /= 255.0
	# TODO: Use "create_multiframe_states" here.
	x = np.zeros((episode.shape[0], history_size, 83, 83))
	for i in range(history_size):
		x[(history_size - i - 1):, i, :, :] = episode[:(episode.shape[0] - history_size + i + 1)]
	y = np.load(episode_path)['y']
	y = y[::4]
	return [x, y]

def load_episode_and_shuffle(database, episode_index, history_size):
	[x, y] = loadEpisode(database, episode_index, history_size)
	indices = numpy.random.permutation(x.shape[0])
	x = x[indices]
	y = y[indices]
	return [x, y]

def sgd_updates_adagrad(cost, params, learning_rate, consider_constant = [], epsilon=1e-10):
	accumulators = OrderedDict({})
	e0s = OrderedDict({})
	learn_rates = []
	ups = OrderedDict({})
	eps = OrderedDict({})
	for param in params:
		eps_p = numpy.zeros_like(param.get_value())
		accumulators[param] = theano.shared(value=eps_p, name="acc_%s" % param.name)
		e0s[param] = learning_rate
		eps_p[:] = epsilon
		eps[param] = theano.shared(value=eps_p, name="eps_%s" % param.name)
	gparams = T.grad(cost, params, consider_constant)
	for param, gp in zip(params, gparams):
		acc = accumulators[param]
		ups[acc] = acc + T.sqr(gp)
		val = T.sqrt(T.sum(ups[acc])) + epsilon
		learn_rates.append(e0s[param] / val)
	updates = [(p, p - step * gp) for (step, p, gp) in zip(learn_rates, params, gparams)]
	return updates

def trainDeepMindNetwork(
	game,
	history_size,
	non_linearity,
	database,
	output_suffix,
	learning_rate=0.1,
	numEpochs=1000,
	batch_size=32,
	maxEpisodes=500,
	use_rica_filters = False):

	game_settings = generalUtils.get_game_settings(game)
	num_actions = len(game_settings.possible_actions)
	[x, y] = load_episode_and_shuffle(database, 0, history_size)

	testSetX = theano.shared(numpy.asarray(x, dtype=theano.config.floatX), borrow=True)
	testSetY = theano.shared(numpy.asarray(y, dtype='int32'), borrow=True)
	[x, y] = load_episode_and_shuffle(database, 1, history_size)
	trainSetX = theano.shared(numpy.asarray(x, dtype=theano.config.floatX), borrow=True)
	trainSetY = theano.shared(numpy.asarray(y, dtype='int32'), borrow=True)
	
	index = T.lscalar() #scalar
	x = T.tensor4('x') #80x80x4xepisodelength/4
	y = T.ivector('y') #vector of ints
	rng = numpy.random.RandomState(23455)
	
	if history_size == 1 and use_rica_filters:
		W0 = np.loadtxt('W.txt', dtype='float32')

		W0_reshaped = np.zeros((16, 1, 8, 8), dtype='float32')
		for i in range(16):
			W0_reshaped[i][0] = W0[i].reshape(8, 8).T

		shared_W0 = theano.shared(W0_reshaped, borrow = True)

		net = DeepMindNetwork(
			x = x,
			num_actions = num_actions,
			batch_size = batch_size,
			rng = rng,
			history_size = history_size,
			non_linearity = non_linearity,
			W0 = shared_W0)

	else:
		net = DeepMindNetwork(
			x = x,
			num_actions = num_actions,
			batch_size = batch_size,
			rng = rng,
			history_size = history_size,
			non_linearity = non_linearity)

	cost = net.layer3.negative_log_likelihood(y)

	params = net.params

	updates = sgd_updates_adagrad(cost = cost, params = params, learning_rate = learning_rate)

	train_model = theano.function([index], net.layer3.errors(y), updates=updates,
		givens={
		x: trainSetX[index * batch_size: (index + 1) * batch_size],
		y: trainSetY[index * batch_size: (index + 1) * batch_size]})
	
	test_model = theano.function([index], net.layer3.errors(y),
		givens={
		x: testSetX[index * batch_size: (index + 1) * batch_size],
		y: testSetY[index * batch_size: (index + 1) * batch_size]})

	output_path = game + '/' + non_linearity + str(history_size) + output_suffix + '/'

	if not exists(game):
		mkdir(game)
	if not exists(output_path):
		mkdir(output_path)
	
	for epochIndex in range(numEpochs):
		all_batch_scores = None
		for episodeIndex in range(1, maxEpisodes):
			if (os.path.isfile(database + str(episodeIndex) + '.npz')):
				print 'episode: ' + str(episodeIndex)
				start = time.time()
				[x, y] = load_episode_and_shuffle(database, episodeIndex, history_size)
				trainSetX.set_value(numpy.asarray(x, dtype=theano.config.floatX))
				trainSetY.set_value(numpy.asarray(y, dtype='int32'))

				end = time.time()
				print 'time for loading episode: ' + str(end - start) + 's'

				start = time.time()
				trainSetSize = trainSetX.get_value(borrow=True).shape[0]

				batch_scores = [train_model(minibatchIndex)
					for minibatchIndex in xrange(trainSetSize / batch_size)]
				if all_batch_scores == None:
					all_batch_scores = batch_scores
				else:
					all_batch_scores = np.append(all_batch_scores, batch_scores)

				end = time.time()
				print 'time for minibatch updates: ' + str(end - start) + 's'
				print ''

		net.saveToFile(output_path + str(epochIndex))

	# Testing saving and loading of the network

	#loadedNetwork = DeepMindNetwork.readFromFile(
	# x = x, num_actions = num_actions, batch_size = batch_size, rng = rng,
	# fileName = 'data/network' + str(epochIndex) + '.npz')
	#print 'comparison of networks: ' + str(DeepMindNetwork.compareNetworkParameters(net, loadedNetwork))
	
	training_score = np.mean(all_batch_scores)
	training_file = open(output_path + 'training.txt', 'a+')
	training_file.write(str(training_score) + '\n')
	training_file.close()

	testSetSize = testSetX.get_value(borrow=True).shape[0]
	validationScore = numpy.mean([test_model(minibatchIndex)
		for minibatchIndex in xrange(testSetSize / batch_size)])
	validation_file = open(output_path + 'validation.txt', 'a+')
	validation_file.write(str(validationScore) + '\n')
	validation_file.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generates gameplay from a given network')
	parser.add_argument('-g','--game', help='Which game', required=True)
	parser.add_argument('-s','--history_size', help='Number of frames in input', required=True, type = int)
	parser.add_argument('-n','--non_linearity', help='tanh or ReLU', required=True)
	parser.add_argument('-d','--database', help='Human gameplay database', required=True)
	parser.add_argument('-o','--output_suffix', help='Suffix for output folder', required=True)

	args = vars(parser.parse_args())


	trainDeepMindNetwork(args['game'], args['history_size'], args['non_linearity'], args['database'], args[
		'output_suffix'])