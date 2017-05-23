import random
import math

# Container for the points
points = []


# Generate N random points
def create_random_points(size):
    global points
    for n in range(size):
        x = random.uniform(0, 500)
        y = random.uniform(0, 500)
        points.append([x, y])


# Find the path through those points
def order_points():
    global points

    # How many points do we have?
    L = len(points)

    # No points at all
    if L < 1:
        return []

    # A single point
    if L == 1:
        return [0]

    # Calculate all the different distances
    distances = [0 for n in range(L * L)]

    for n in range(L):
        for m in range(n):
            A = points[n]
            B = points[m]
            distances[n * L + m] = math.sqrt((B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2)
            distances[m * L + n] = distances[n * L + m]

    # Calculate the best path: dynamic programming
    # Initialise the distance and path variables
    sum_path = [0 for n in range(L - 1)]
    path = [[0] for n in range(L - 1)]

    # Calculate the first iteration
    for point_m in range(1, L):
        sum_path[point_m - 1] += distances[point_m]
        path[point_m - 1].append(point_m)

    # Calculate the following iterations
    for n in range(1, L - 1):
        for point_m in range(1, L):
            dist = -1
            prev = -1
            for point_n in range(1, L):
                if point_n == point_m:
                    continue
                if point_n in path[point_m - 1]:
                    continue

                d = distances[point_m * L + point_n]
                if dist == -1 or dist < d:
                    dist = d
                    prev = point_n
            sum_path[point_m - 1] += dist
            path[point_m - 1].append(prev)

    # Calculate the last iteration
    for point_m in range(1, L):
        sum_path[point_m - 1] += distances[point_m]
        path[point_m - 1].append(0)

    best_score = min(sum_path)
    for point_m in range(1, L):
        if sum_path[point_m - 1] == best_score:
            best_path = path[point_m - 1]

    return best_path

create_random_points(5)

print ('We have the following points:')
print (points)

print ('')
print ('And the best path is:')
print (order_points())