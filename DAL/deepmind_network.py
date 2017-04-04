import numpy as np
from conv_layer import LeNetConvPoolLayer
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
import theano.tensor as T
import theano

def relu(x):
	return T.switch(T.lt(x, 0), 0, x)

def identity(x):
	return x

class DeepMindNetwork:

	HIDDEN_LAYER_SIZE = 256
	
	def __init__(self, x, num_actions, batch_size, rng, history_size, non_linearity, W0 = None, b0 = None, W1 = None, b1 = None, W2 = None, b2 = None, W3 = None, b3 = None):
		self.layer0 = LeNetConvPoolLayer(
			rng = rng,
			input = x,
			filter_shape = (16, history_size, 8, 8),
			image_shape = (batch_size, history_size, 83, 83),
			poolsize = (4, 4),
			W = W0,
			b = b0,
			relu = (non_linearity == 'relu'))
		
		self.layer1 = LeNetConvPoolLayer(
			rng = rng,
			input = self.layer0.output,
			filter_shape = (32, 16, 4, 4),
			image_shape = (batch_size, 16, 19, 19),
			poolsize = (2, 2),
			W = W1,
			b = b1,
			relu = (non_linearity == 'relu'))
		
		layer2_input = self.layer1.output.flatten(2)
		
		if non_linearity == 'relu':
			activation = relu
		else:
			activation = T.tanh
		
		self.layer2 = HiddenLayer(
			rng = rng,
			input = layer2_input,
			n_in = 2048,
			n_out = DeepMindNetwork.HIDDEN_LAYER_SIZE,
			activation = activation,
			W = W2,
			b = b2)

		self.layer3 = LogisticRegression(
			input = self.layer2.output,
			n_in = DeepMindNetwork.HIDDEN_LAYER_SIZE,
			n_out = num_actions,
			W = W3,
			b = b3)

		self.params = self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params

	def saveToFile(self, fileName):
		np.savez_compressed(
		fileName,
		W0 = self.layer0.W.get_value(),
		b0 = self.layer0.b.get_value(),
		W1 = self.layer1.W.get_value(),
		b1 = self.layer1.b.get_value(),
		W2 = self.layer2.W.get_value(),
		b2 = self.layer2.b.get_value(),
		W3 = self.layer3.W.get_value(),
		b3 = self.layer3.b.get_value())
	
	@staticmethod
	def readFromFile(x, num_actions, batch_size, rng, fileName, history_size, non_linearity):
		W0 = np.load(fileName)['W0']
		b0 = np.load(fileName)['b0']
		W1 = np.load(fileName)['W1']
		b1 = np.load(fileName)['b1']
		W2 = np.load(fileName)['W2']
		b2 = np.load(fileName)['b2']
		W3 = np.load(fileName)['W3']
		b3 = np.load(fileName)['b3']
		return DeepMindNetwork(
			x, num_actions, batch_size, rng, history_size, non_linearity,
			W0 = theano.shared(np.asarray(W0, dtype=theano.config.floatX), borrow=True),
			b0 = theano.shared(np.asarray(b0, dtype=theano.config.floatX), borrow=True),
			W1 = theano.shared(np.asarray(W1, dtype=theano.config.floatX), borrow=True),
			b1 = theano.shared(np.asarray(b1, dtype=theano.config.floatX), borrow=True),
			W2 = theano.shared(np.asarray(W2, dtype=theano.config.floatX), borrow=True),
			b2 = theano.shared(np.asarray(b2, dtype=theano.config.floatX), borrow=True),
			W3 = theano.shared(np.asarray(W3, dtype=theano.config.floatX), borrow=True),
			b3 = theano.shared(np.asarray(b3, dtype=theano.config.floatX), borrow=True))

	@staticmethod
	def compareNetworkParameters(net1, net2):
		if not np.array_equal(net1.layer0.W.get_value(), net2.layer0.W.get_value()):
			return False
		if not np.array_equal(net1.layer0.b.get_value(), net2.layer0.b.get_value()):
			return False
		if not np.array_equal(net1.layer1.W.get_value(), net2.layer1.W.get_value()):
			return False
		if not np.array_equal(net1.layer1.b.get_value(), net2.layer1.b.get_value()):
			return False
		if not np.array_equal(net1.layer2.W.get_value(), net2.layer2.W.get_value()):
			return False
		if not np.array_equal(net1.layer2.b.get_value(), net2.layer2.b.get_value()):
			return False
		if not np.array_equal(net1.layer3.W.get_value(), net2.layer3.W.get_value()):
			return False
		if not np.array_equal(net1.layer3.b.get_value(), net2.layer3.b.get_value()):
			return False
		return True