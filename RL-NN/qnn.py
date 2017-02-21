from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback
from keras.layers.advanced_activations import LeakyReLU



def createModelRelu():

    model = Sequential()
    model.add(Dense(164, init='lecun_uniform', input_shape=(64,)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

    model.add(Dense(150, init='lecun_uniform'))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))		

    model.add(Dense(4, init='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)
    
    return model
	

def createModelLeakyRelu():

	model = Sequential()
	model.add(Dense(164, init='lecun_uniform', input_shape=(64,)))
	model.add(LeakyReLU(alpha=0.01))
	#model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

	model.add(Dense(150, init='lecun_uniform'))
	model.add(LeakyReLU(alpha=0.01))
	#model.add(Dropout(0.2))		

	model.add(Dense(4, init='lecun_uniform'))
	model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

	rms = RMSprop()
	model.compile(loss='mse', optimizer=rms)
    
	return model
	
	