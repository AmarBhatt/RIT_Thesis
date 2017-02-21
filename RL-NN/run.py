#import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'

#import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

from IPython.display import clear_output
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import History 
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.callbacks import LambdaCallback
import random
from environment import *
from qnn import *
import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
import sys


def testAlgo(init, thefile, model):
    i = 0
    print("Initial State:")
    if init==0:
        state = initGrid()
    elif init==1:
        state = initGridPlayer()
    elif init==2:
        state = initGridRand()
    elif init==3:
        state = initGridRandPlayer()

    grid = dispGrid(state);
    
    print(grid)

    status = 1
    #while game still in progress
    while(status == 1):
        qval = model.predict(state.reshape(1,64), batch_size=1)
        action = (np.argmax(qval)) #take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action),flush=True)
        #thefile.writelines('Move #: %s; Taking action: %s\n' % (i, action))
        state = makeMove(state, action)
        #print(dispGrid(state))
        reward = getReward(state)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
            if reward == 10:
                return 1,reward,i
            else:
                return 0,reward,i
            #thefile.writelines("Reward: %s\n" % (reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 10):
            print("Game lost; too many moves.")
            #thefile.writelines("Game lost; too many moves.\n")
            return 0,0,100
            break

def testBed(init,num,epochs, model):
    winloss = [];
    reward = [];
    steps = [];
    wins = 0;
    lose = 0;
    unfinished = 0;
    #thefile = open(str(init)+'-test.txt', 'w')
    for i in range(num):
        print("*-*-*-*-* Test %s *-*-*-*-*"%(i,),flush=True)
        #thefile.writelines("*-*-*-*-* Test %s *-*-*-*-*\n"%(i,))
        g,r,s = testAlgo(init,"thefile", model)
        #winloss.append(g)
        #reward.append(r)
        #steps.append(s)
        if(r == 10):
            wins+=1
        if(r == -10):
            lose+=1
        if(r == 0):
            unfinished+=1
    print("Wins/Game: %s/%s"%(wins,num),flush=True)
    print("Lose/Game: %s/%s"%(lose,num),flush=True)
    print("Unfinished/Game: %s/%s"%(unfinished,num),flush=True)
    testFile.write(str(epochs) + ", " + str(num) + ", " + str(wins) + ", " + str(lose) + ", " + str(unfinished) + ", " + str(s) + "\n") #episode, test amount, win, lose, unfinished, steps

def startTrainingWithExperienceReplay(stateType,epochs,gamma,epsilon,batchSize,buffer,model,testInterval,numTests):
    replay = [] #stores tuples of (S, A, R, S')
    h = 0
    loss = [];
    timeTot = 0
    last_min = 0
    last_max = 2
    
    for i in range(epochs):
        if stateType==0:
            state = initGrid()
        elif stateType==1:
            state = initGridPlayer()
        elif stateType==2:
            state = initGridRand()
        elif stateType == 3:
            state = initGridRandPlayer()
        status = 1
        print("Game %s/%s" % (i,epochs))
        #while game still in progress
        start = time.clock()
        while(status == 1):
            #We are in state S
            #Let's run our Q function on S to get Q values for all possible actions
            qval = model.predict(state.reshape(1,64), batch_size=1)
            if (random.random() < epsilon): #choose random action
                action = np.random.randint(0,4)
            else: #choose best action from Q(s,a) values
                action = (np.argmax(qval))
            #Take action, observe new state S'
            new_state = makeMove(state, action)
            #Observe reward
            reward = getReward(new_state)

            #Experience replay storage
            if (len(replay) < buffer): #if buffer not filled, add to it
                replay.append((state, action, reward, new_state))
            else: #if buffer full, overwrite old values
                if (h < (buffer-1)):
                    h += 1
                else:
                    h = 0
                replay[h] = (state, action, reward, new_state)
                #randomly sample our experience replay memory
                minibatch = random.sample(replay, batchSize)
                X_train = []
                y_train = []
                for memory in minibatch:
                    #Get max_Q(S',a)
                    old_state_m, action_m, reward_m, new_state_m = memory
                    old_qval = model.predict(old_state_m.reshape(1,64), batch_size=1)
                    newQ = model.predict(new_state_m.reshape(1,64), batch_size=1)
                    maxQ = np.max(newQ)
                    y = np.zeros((1,4))
                    y[:] = old_qval[:]
                    if reward_m == -1: #non-terminal state
                        update = (reward_m + (gamma * maxQ))
                    else: #terminal state
                        update = reward_m
                    y[0][action_m] = update
                    X_train.append(old_state_m.reshape(64,))
                    y_train.append(y.reshape(4,))

                X_train = np.array(X_train)
                y_train = np.array(y_train)
                hist = model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=0)
                loss.append(hist.history['loss'][0])

                state = new_state
            if reward != -1: #if reached terminal state, update game status
                status = 0
                end = time.clock()
                timeTot +=(end-start)
            clear_output(wait=True)
        if epsilon > 0.01: #decrement epsilon over time, used to be 0.1 or 10%
            epsilon -= (1/epochs)
        print("Time Elapsed: %f"%(end-start,))
        print("Avg. Time/Game: %f"%(timeTot/(i+1),))
        if len(loss) > 0:
            print("Loss: %f"%(loss[-1],),flush=True)
            lossFile.write(str(i)+", "+str(loss[-1]) + "\n")
        if(i%testInterval == 0):
            model.save(filename+"/models/"+str(stateType)+'-model-'+str(i)+'.h5')
            testBed(stateType,numTests,i, model)
            fig = plt.figure()
            plt.ion()
            ax = plt.gca()
            ax.set_autoscale_on(True)
            line, = ax.plot([], [])
            plt.ylabel('error')
            plt.xlabel('iterations over time')
            plt.title('loss over time')
            line.set_xdata(np.append(line.get_xdata(), range(0,len(loss))))
            line.set_ydata(np.append(line.get_ydata(), loss))
            ax.relim()
            ax.autoscale_view(True,True,True)
            plt.draw()
            plt.pause(0.000000000000000001)
            fig.savefig(filename+"/images/"+str(stateType)+'-loss-'+str(i)+'.png', bbox_inches='tight')
            plt.close(fig)
    
    lossFile.write(str(i)+", "+str(loss[-1]) + "\n")
    model.save(filename+"/models/"+str(stateType)+'-model-'+str(i)+'.h5')
    testBed(stateType,numTests,i, model)
    fig = plt.figure()
    plt.ion()
    ax = plt.gca()
    ax.set_autoscale_on(True)
    line, = ax.plot([], [])
    plt.ylabel('error')
    plt.xlabel('iterations over time')
    plt.title('loss over time')
    line.set_xdata(np.append(line.get_xdata(), range(0,len(loss))))
    line.set_ydata(np.append(line.get_ydata(), loss))
    ax.relim()
    ax.autoscale_view(True,True,True)
    plt.draw()
    plt.pause(0.000000000000000001)
    fig.savefig(filename+"/images/"+str(stateType)+'-loss-'+str(i)+'.png', bbox_inches='tight')
    plt.close(fig)
    return loss   


def main (stateType,epochs,gamma,epsilon,batchSize,buffer, testInterval, numTests, filename):

    for i in range(0,len(stateType)):
        model = createModelRelu()
        name = ""
        st = stateType[i]
        if st == 0:
            name = "All Stationary"
        elif st == 1:
            name = "Map Stationary"
        elif st == 2:
            name = "None Stationary"
        elif st == 3:
            name = "Player Stationary"
        print("################### BEGIN %s #############################" % (name,))
        loss = startTrainingWithExperienceReplay(stateType[i],epochs[i],gamma,epsilon,batchSize,buffer,model,testInterval[i],numTests[i])
    

#print(model.predict(state.reshape(1,64), batch_size=1))
#just to show an example output; read outputs left to right: up/down/left/right
'''
epochs = 50000 #number of games, 3000
gamma = 0.975 #discount factor
epsilon = 1 #policy
batchSize = 40 #mini-batch ammount - used to be 40
buffer = 80 #experience replay size - used to be 80
stateType = 3; #0 - stationary, 1- random player, 2-random environment and random player, 3- random environment
numTests = 5000;


epochsList = [5000];#[50000,50000,100000,100000];
stateTypeList = [1]; #[0,1,3,2]
testInterval = [10]; #[5000, 5000, 10000, 10000];
numTestsList = [500]; #[5000, 5000, 10000, 10000];
'''
'''
epochsList = [5,5,10,10];
stateTypeList = [0,1,3,2]
testInterval = [2, 2, 5, 5];
numTestsList = [2, 2, 5, 5];
'''

import os

#dir is not keyword
def makemydir(filename):
  try:
    os.makedirs(filename)
  except OSError:
    pass

if __name__ == "__main__":
    print("Begin.")
    #print(sys.argv)
    filename = sys.argv[7]
    makemydir(filename+"/models")
    makemydir(filename+"/images")
    makemydir(filename)
    test = filename+"/test-"+filename+".csv" #episode, test amount, win, lose, unfinished, steps
    losscost = filename+"/loss-"+filename+".csv" #episode, cost
    all = filename+"/"+filename+".txt" #everything (piped)
    # open files
    testFile = open(test, 'w')
    testFile.write("episode,num tests, win, lose, unfinished, steps\n")
    lossFile = open(losscost, 'w')
    lossFile.write("episode, cost\n")
    stateTypeList = [int(s) for s in sys.argv[1].split(',')]
    epochsList = [int(s) for s in sys.argv[2].split(',')]
    gamma = 0.975
    epsilon = 1
    batchSize = int(sys.argv[3])
    buffer = int(sys.argv[4])
    testInterval = [int(s) for s in sys.argv[5].split(',')]
    numTestsList = [int(s) for s in sys.argv[6].split(',')]
    main(stateTypeList,epochsList,gamma,epsilon,batchSize,buffer,testInterval, numTestsList, filename)
    #close files
    testFile.close()
    lossFile.close()
    # Read in the file
    filedata = None
    with open(all, 'r') as file :
      filedata = file.read()
    # Replace the target string
    filedata = filedata.replace('[2K', '')
    file.close()
    # Write the file out again
    with open(all, 'w') as file:
      file.write(filedata)
    file.close()
