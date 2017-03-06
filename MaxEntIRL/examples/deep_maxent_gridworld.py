"""
Run maximum entropy inverse reinforcement learning on the objectworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt

import irl.deep_maxent as deep_maxent
import irl.mdp.gridworld_flex as gridworld
import irl.value_iteration as value_iteration

from irl.mdp.imageHandler import *

def main(grid_size, discount, n_trajectories, epochs, learning_rate, obstacle_list, pit_list, goal, wind, structure):
    """
    Run deep maximum entropy inverse reinforcement learning on the objectworld
    MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_objects: Number of objects. int.
    n_colours: Number of colours. int.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    structure: Neural network structure. Tuple of hidden layer dimensions, e.g.,
        () is no neural network (linear maximum entropy) and (3, 4) is two
        hidden layers with dimensions 3 and 4.
    """
    print("START!");
    #wind = 0.3
    trajectory_length = 3*grid_size #8
    l1 = l2 = 0

    gw = gridworld.Gridworld(grid_size, wind, discount,obstacle_list,pit_list,goal)
    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    #policy = find_policy(gw.n_states, gw.n_actions, gw.transition_probability,
    #                     ground_r, ow.discount, stochastic=False)
    trajectories = gw.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            gw.optimal_policy, True)
    feature_matrix = gw.feature_matrix("ident")
    r,_ = deep_maxent.irl((feature_matrix.shape[1],) + structure, feature_matrix,
        gw.n_actions, discount, gw.transition_probability, trajectories, epochs,
        learning_rate, l1=l1, l2=l2)

    #print(ground_r);
    #print("******************");
    #print(r);
    #print("*******************");
    #print(np.fabs(ground_r-r));

    # plt.subplot(1, 2, 1)
    # plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    # plt.colorbar()
    # plt.title("Groundtruth reward")
    # plt.subplot(1, 2, 2)
    # plt.pcolor(r.reshape((grid_size, grid_size)))
    # plt.colorbar()
    # plt.title("Recovered reward")
    # plt.show()

    policy = value_iteration.find_policy(grid_size**2, 4,
                                         gw.transition_probability, r, discount)
    #print(policy)
    policy_map = np.chararray((grid_size, grid_size))
    actions = [r'$\rightarrow$',r'$\downarrow$',r'$\leftarrow$',r'$\uparrow$']
    count = 0
    for s in policy:
        sx, sy = gw.int_to_point(count)
        if (count not in  obstacle_list):
            a = np.argmax(s)
            policy_map[sy,sx] = actions[a]
        else:
            policy_map[sy,sx] = 'X'
            r[count] = np.amin(r); 
        if count == goal:
            policy_map[sy,sx] = 'G'
        count += 1
    #print(policy_map)

    print("Plotting")
    
    plt.figure() 
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    plt.pcolor(ground_r.reshape((grid_size, grid_size)),cmap='gray')
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.gca().invert_yaxis()
    ax.grid(color='k', linestyle='-', linewidth=2, axis='both')
    plt.subplot(1, 2, 2)
    ax = plt.gca()
    plt.pcolor(r.reshape((grid_size, grid_size)),cmap='gray')
    count = 0
    for y in range(grid_size):
        for x in range(grid_size):
            a = np.argmax(policy[count])
            if(count not in obstacle_list):
                plt.text(x + 0.5, y + 0.5, '%s' % actions[a],
                         horizontalalignment='center',
                         verticalalignment='center', color='orange',fontsize='15',
                         )
            else:
                plt.text(x + 0.5, y + 0.5, '%s' % 'X',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='black',)
            count += 1

    plt.colorbar()
    plt.title("Recovered reward")
    plt.gca().invert_yaxis()
    ax.grid(color='k', linestyle='-', linewidth=2)
    plt.show()
    print("Done")

if __name__ == '__main__':
    grid = readImage("128x128.png")
    obstacles = []
    count = 0
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if(grid[y,x] == 0):
                obstacles.append(count)
            count += 1
    grid_size = grid.shape
    print(obstacles)
    main(grid_size[0], 0.9, 20, 200, 0.01,obstacles,[],8492, 0.0, (3, 3)) #500000
