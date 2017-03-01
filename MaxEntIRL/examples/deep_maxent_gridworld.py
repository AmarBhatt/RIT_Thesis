"""
Run maximum entropy inverse reinforcement learning on the objectworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt

import irl.deep_maxent as deep_maxent
import irl.mdp.gridworld_flex as gridworld
from irl.value_iteration import find_policy

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
    r = deep_maxent.irl((feature_matrix.shape[1],) + structure, feature_matrix,
        gw.n_actions, discount, gw.transition_probability, trajectories, epochs,
        learning_rate, l1=l1, l2=l2)

    print(ground_r);
    print("******************");
    print(r);
    print("*******************");
    print(np.fabs(ground_r-r));

    plt.subplot(1, 2, 1)
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()

if __name__ == '__main__':
    main(10, 0.9, 20, 200, 0.01,[],[],24, 0.3, (3, 3)) #500000
