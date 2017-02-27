"""
Run maximum entropy inverse reinforcement learning on the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt

import irl.maxent as maxent
import irl.mdp.gridworld_flex as gridworld
import irl.value_iteration as value_iteration

def main(grid_size, discount, n_trajectories, epochs, learning_rate, obstacle_list, goal, wind):
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """

    #wind = 0.3 #0.3
    trajectory_length = 3*grid_size

    print("Setting up GridWorld")

    gw = gridworld.Gridworld(grid_size, wind, discount,obstacle_list,goal)
    trajectories = gw.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            gw.optimal_policy, True)
    
    #print(trajectories)
    feature_matrix = gw.feature_matrix('ident')
    #print(feature_matrix)
    #print(gw.transition_probability)
    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    #print(gw.transition_probability)
    print("Running Maximum Entropy IRL")

    r, goe,policy = maxent.irl(feature_matrix, gw.n_actions, discount,
        gw.transition_probability, trajectories, epochs, learning_rate)
    #print(r)

    print(policy)
    policy_map = np.chararray((grid_size, grid_size))
    actions = ['>','v','<','^']
    count = 0
    for s in policy:
        sx, sy = gw.int_to_point(count)
        if (count not in  obstacle_list):
            a = np.argmax(s)
            policy_map[sy,sx] = actions[a]
        else:
            policy_map[sy,sx] = 'X'
        if count == goal:
            policy_map[sy,sx] = 'G'
        count += 1
    print(policy_map)

    print("Plotting")
    
    plt.figure() 
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.gca().invert_yaxis()
    ax.grid(color='k', linestyle='-', linewidth=2, axis='both')
    plt.subplot(1, 2, 2)
    ax = plt.gca()
    plt.pcolor(r.reshape((grid_size, grid_size)))
    count = 0
    for y in range(grid_size):
        for x in range(grid_size):
            a = np.argmax(policy[count])
            if(count not in obstacle_list):
                plt.text(x + 0.5, y + 0.5, '%s' % actions[a],
                         horizontalalignment='center',
                         verticalalignment='center',
                         )
            else:
                plt.text(x + 0.5, y + 0.5, '%s' % 'X',
                         horizontalalignment='center',
                         verticalalignment='center',
                         )
            count += 1

    plt.colorbar()
    plt.title("Recovered reward")
    plt.gca().invert_yaxis()
    ax.grid(color='k', linestyle='-', linewidth=2)
    plt.show()
    print("Done")

    #policy = value_iteration.find_policy(grid_size, 4,
                                         #gw.transition_probability, r, discount)
    



if __name__ == '__main__':
    main(10, 0.01, 20, 200, 0.01, [15,45,63,65,72,24], 64, 0.3) 
    #3,17,11,7S

#policy map
#obstacles at negative rewards