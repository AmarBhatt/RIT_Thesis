import numpy as np

def optimal_policy(grid_size, goal):
	
	#print(create_graph(w))
	return


def int_to_point(i, grid_size):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """

        return (i % grid_size, i // grid_size)

def point_to_int(p, grid_size):
    """
    Convert a coordinate into the corresponding state int.

    p: (x, y) tuple.
    -> State int.
    """

    return p[0] + p[1]*grid_size

def create_world(grid_size, obstacle_list, goal):
	# Generate world
	w = np.zeros(shape=(grid_size,grid_size))
	for obstacle in obstacle_list:
		(x,y) = int_to_point(obstacle, grid_size)
		w[y,x] = 1
	(x,y) = int_to_point(goal, grid_size)
	w[y,x] = 2
	return w


def create_graph(world):
	graph = {}

	for y in range(world.shape[0]):
		for x in range(world.shape[1]):
			neighbor_list = [];
			if not(world[y,x] == 1):
				if x+1 < len(world) and not(world[y,x+1] == 1):
					neighbor_list.append(point_to_int((x+1,y),len(world)))
				if y+1 < len(world) and not(world[y+1,x] == 1):
					neighbor_list.append(point_to_int((x,y+1),len(world)))
				if x-1 >= 0 and not(world[y,x-1] == 1):
					neighbor_list.append(point_to_int((x-1,y),len(world)))
				if y-1 >= 0 and not(world[y-1,x] == 1):
					neighbor_list.append(point_to_int((x,y-1),len(world)))			
			
			graph[point_to_int((x,y),len(world))] = neighbor_list;
	return graph




def bfs(graph, start, end):
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
        # enumerate all adjacent nodes, construct a new path and push it into the queue
        for adjacent in graph.get(node, []):
            new_path = list(path)
            new_path.append(adjacent)
            queue.append(new_path)
w = create_world(5,[3,11,14,23], 12)
graph = create_graph(w)
print (bfs(graph, 24, 24))