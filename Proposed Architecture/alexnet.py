# -*- coding: utf-8 -*-

""" AlexNet.
Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.
References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from DataSetGenerator.maze_generator import *
#import tflearn.datasets.oxflower17 as oxflower17

import os.path
import h5py
#X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

f_data = "random224_8"
f_file = "random224_8.h5"
size = 224

num = 80
num2 = 100

if not os.path.isfile(f_file):

    episode_lengths = []
    episode_start = []
    episode_total = 0
    image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(0),size)

    x_train = image_set
    y_train = action_set

    episode_lengths.append(action_set.shape[0])
    episode_start.append(episode_total)
    episode_total += image_set.shape[0]
    for i in range(1,num):
        print(i)
        image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(i),size)
        x_train = np.append(x_train,image_set,axis=0)
        y_train = np.append(y_train,action_set, axis=0)
        episode_lengths.append(action_set.shape[0])
        episode_start.append(episode_total)
        episode_total += image_set.shape[0]
    #print(episode_lengths)
    print(x_train.shape)
    print(y_train.shape)

    image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(num),size)

    x_test = image_set
    y_test = action_set

    for i in range(num+1,num2):
        image_set,action_set = processGIF('DataSetGenerator/expert_data/'+f_data+"/"+str(i),size)
        x_test = np.append(x_test,image_set,axis=0)
        y_test = np.append(y_test,action_set, axis=0)

    print(x_test.shape)
    print(y_test.shape)
    
    with h5py.File(f_file,'w') as hf:
        hf.create_dataset("x_train",data=x_train, shape = x_train.shape)
        hf.create_dataset("y_train",data=y_train, shape = y_train.shape)
        hf.create_dataset("episode_lengths",data=episode_lengths)
        hf.create_dataset("episode_start",data=episode_start)
        hf.create_dataset("episode_total",data = [episode_total])
        hf.create_dataset("x_test",data=x_test, shape=x_test.shape)
        hf.create_dataset("y_test",data=y_test, shape=y_test.shape)
else:
    with h5py.File(f_file, 'r') as hf:
        x_train = hf['x_train'][:]
        y_train = hf['y_train'][:]
        episode_lengths = hf['episode_lengths'][:]
        episode_start = hf['episode_lengths'][:]
        episode_total = hf['episode_total'][:]
        episode_total = episode_total[0]
        x_test = hf['x_test'][:]
        y_test = hf['y_test'][:]



# Building 'AlexNet'
network = input_data(shape=[None, 224, 224, 1])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
model.fit(x_train, y_train, n_epoch=1000, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_oxflowers17')