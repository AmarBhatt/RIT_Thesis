"""
Implements the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import numpy.random as rn
from collections import deque
import networkx as nx

class Gridworld(object):
    """
    Gridworld MDP.
    """

    def __init__(self, grid_size, wind, obstacles, goal):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Gridworld
        """

        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1)) #right, down, left, up
        self.n_actions = len(self.actions)
        self.n_states = grid_size**2
        self.grid_size = grid_size
        self.wind = wind
        #self.discount = discount
        self.goal = goal
        self.obstacle_list = obstacles
        #self.pit_list = pits
        self.world_grid = self.create_world()
        #print("Set up graph")
        self.graph = self.create_graph()
        #print("Get shortest paths")
        self.paths = self.get_all_shortest_paths(self.goal)
        #print(self.world_grid)
        #print(self.graph)
        # Preconstruct the transition probability array.
        #print("Get transition probabilities")
        # self.transition_probability = np.array(
        #     [[[self._transition_probability(i, j, k)
        #        for k in range(self.n_states)]
        #       for j in range(self.n_actions)]
        #      for i in range(self.n_states)])

    def __str__(self):
        return "Gridworld({}, {}, {})".format(self.grid_size, self.wind,
                                              self.discount)

    
    def create_world(self):
        # Generate world
        w = np.zeros(shape=(self.grid_size,self.grid_size))
        for obstacle in self.obstacle_list:
            (x,y) = self.int_to_point(obstacle)
            w[y,x] = 1
        (x,y) = self.int_to_point(self.goal)
        w[y,x] = 2
        return w

    def create_graph(self):
        graph = nx.Graph() #{}
        world = self.world_grid
        for y in range(world.shape[0]):
            for x in range(world.shape[1]):
                neighbor_list = [];
                node = self.point_to_int((x,y));
                if not(world[y,x] == 1):
                    if x+1 < len(world) and not(world[y,x+1] == 1):
                        neighbor_list.append((node,self.point_to_int((x+1,y))))#(self.point_to_int((x+1,y)))
                    if y+1 < len(world) and not(world[y+1,x] == 1):
                        neighbor_list.append((node,self.point_to_int((x,y+1))))#(self.point_to_int((x,y+1)))
                    if x-1 >= 0 and not(world[y,x-1] == 1):
                        neighbor_list.append((node,self.point_to_int((x-1,y))))#(self.point_to_int((x-1,y)))
                    if y-1 >= 0 and not(world[y-1,x] == 1):
                        neighbor_list.append((node,self.point_to_int((x,y-1))))#(self.point_to_int((x,y-1)))          
                
                graph.add_edges_from(neighbor_list)#graph[self.point_to_int((x,y))] = neighbor_list;
        return graph

    def bfs1(self, start, end):
        # maintain a queue of paths
        queue = []
        # push the first path into the queue
        queue.append([start])
        while queue:
            # get the first path from the queue
            path = queue.pop(0)
            # get the last node from the path
            node = path[-1]
            # path found
            if node == end:
                return path
            #if node in self.pit_list:
            #    return path
            # enumerate all adjacent nodes, construct a new path and push it into the queue
            for adjacent in self.graph.get(node, []):
                new_path = list(path)
                new_path.append(adjacent)
                queue.append(new_path)



    def get_all_shortest_paths(self,source):
        return nx.single_source_shortest_path(self.graph,source)

    def bfs(self,start,end):
        return list(reversed(self.paths[start]))





    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """

        return (i % self.grid_size, i // self.grid_size)

    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.

        p: (x, y) tuple.
        -> State int.
        """

        return p[0] + p[1]*self.grid_size

    def neighbouring(self, i, k):
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        i: (x, y) int tuple.
        k: (x, y) int tuple.
        -> bool.
        """

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1

    

    def reward(self, state_int):
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """

        if state_int == self.goal: #// 2: 
            return 1
        elif state_int in self.pit_list:
            return -1
        return 0 #if this is negative it may work better


    def optimal_policy(self, state_int):
        """
        The optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        #get best path
        path = self.bfs(state_int, self.goal)
        #print(state_int)
        #print(path)
        (xi,yi) = self.int_to_point(state_int)
        if len(path) == 1: #at goal
            return rn.randint(0, 4)
        else:
            (xp,yp) = self.int_to_point(path[1])
        
        if xi+1 == xp: #go right
            return 0
        if xi-1 == xp: #go left
            return 2
        if yi+1 == yp: #go down
            return 1
        if yi-1 == yp: #go up
            return 3
        raise ValueError("Unexpected state.")


    def generate_trajectories(self, n_trajectories, trajectory_length, policy,
                                    random_start=False):
        """
        Generate n_trajectories trajectories with length trajectory_length,
        following the given policy.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float)]]
        """

        trajectories = []
        for tracount in range(n_trajectories):
            if random_start:
                sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
                p = self.point_to_int((sx,sy))
                #print(p)
                while(p in self.obstacle_list):
                    sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
                    p = self.point_to_int((sx,sy))
                    #print(p)
            else:
                sx, sy = 0, 0

            trajectory = []
            for _ in range(trajectory_length):
                if rn.random() < self.wind:
                    action = self.actions[rn.randint(0, 4)]
                else:
                    # Follow the given policy.
                    action = self.actions[policy(self.point_to_int((sx, sy)))]

                if ((0 <= sx + action[0] < self.grid_size and
                        0 <= sy + action[1] < self.grid_size) and not (self.world_grid[sy+action[1],sx+action[0]] == 1)):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy

                state_int = self.point_to_int((sx, sy))
                action_int = self.actions.index(action)
                next_state_int = self.point_to_int((next_sx, next_sy))
                reward = self.reward(next_state_int)
                trajectory.append((state_int, action_int, reward))

                sx = next_sx
                sy = next_sy

            trajectories.append(trajectory)
            print(tracount)
        return np.array(trajectories)