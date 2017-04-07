# Random Maze Generator using Depth-first Search
# http://en.wikipedia.org/wiki/Maze_generation_algorithm
# FB - 20121214
import random
from PIL import Image
import grid_world as gridworld

def generate():
    imgx = 100; imgy = 100
    image = Image.new("L", (imgx, imgy))
    pixels = image.load()
    mx = 10; my = 10 # width and height of the maze
    maze = [[0 for x in range(mx)] for y in range(my)]
    dx = [0, 1, 0, -1]; dy = [-1, 0, 1, 0] # 4 directions to move in the maze
    #color = [(0,0, 0), (255, 255, 255), (255,255,0)] # RGB colors of the maze
    color = [0,255,128]
    # start the maze from a random cell
    stack = [(random.randint(0, mx - 1), random.randint(0, my - 1))]

    while len(stack) > 0:
        (cx, cy) = stack[-1]
        maze[cy][cx] = 1
        # find a new cell to add
        nlst = [] # list of available neighbors
        for i in range(4):
            nx = cx + dx[i]; ny = cy + dy[i]
            if nx >= 0 and nx < mx and ny >= 0 and ny < my:
                if maze[ny][nx] == 0:
                    # of occupied neighbors must be 1
                    ctr = 0
                    for j in range(4):
                        ex = nx + dx[j]; ey = ny + dy[j]
                        if ex >= 0 and ex < mx and ey >= 0 and ey < my:
                            if maze[ey][ex] == 1: ctr += 1
                    if ctr == 1: nlst.append(i)
        # if 1 or more neighbors available then randomly select one and move
        if len(nlst) > 0:
            ir = nlst[random.randint(0, len(nlst) - 1)]
            cx += dx[ir]; cy += dy[ir]
            stack.append((cx, cy))
        else: stack.pop()

    # paint the maze
    for ky in range(imgy):
        for kx in range(imgx):
            pixels[kx, ky] = color[maze[my * ky // imgy][mx * kx // imgx]]
    ky = random.randint(0,imgy-1)
    kx = random.randint(0,imgx-1)

    while(maze[my * ky // imgy][mx * kx // imgx] != 1):
        ky = random.randint(0,imgy-1)
        kx = random.randint(0,imgx-1)

    countup = 0
    countdown = 0
    countleft = 0
    countright = 0

    #up
    kyy=ky-1
    while (kyy > -1 and maze[my * kyy // imgy][mx * kx // imgx] != 0 and (ky-countup)%my !=0):
        countup += 1
        kyy-=1
        print(ky)
    #left
    kxx=kx-1
    while (kxx > -1 and maze[my * ky // imgy][mx * kxx // imgx] != 0 and (kx-countleft)%mx !=0):
        countleft += 1
        kxx-=1
        print(kx)
    print("Paint!")
    for i in range(ky-countup,(ky-countup)+imgy//my):
        for j in range(kx-countleft,(kx-countleft)+imgx//mx):
            pixels[j, i] = color[2]
            print((j,i))
    maze[my * ky // imgy][mx * kx // imgx] = 2

    image.save("Maze_" + str(mx) + "x" + str(my) + ".png", "PNG")
    
    return maze, image



def find_obstacles(maze):
    obstacles = []
    goal = 0
    count = 0
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if(maze[y][x] == 0):
                obstacles.append(count)
            if(maze[y][x] == 2):
                goal = count
            count += 1

    return obstacles, goal


def createTrajectories(grid_size, n_trajectories, obstacle_list, goal, wind):

    #wind = 0.3
    trajectory_length = 3*grid_size #8

    gw = gridworld.Gridworld(grid_size, wind, discount,obstacle_list,pit_list,goal)

    trajectories = gw.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            gw.optimal_policy, True)

    return trajectories, gw



def createDataSet(name,gw,maze,image, mx, my):

    start = random.randint(0,gw.grid_size**2 - 1)
    sx,sy = gw.int_to_point(start)
    while(gw.world_grid[sy][sx] > 0):
        start = random.randint(0,gw.grid_size**2 - 1)
        sx,sy = gw.int_to_point(start)
    path = gw.bfs(start,gw.goal)
    count = 0
    color = 64
    for p in path:
        i_tmp = image.copy()
        pixels = i_tmp.load()
        sx,sy = gw.int_to_point(p)
        print(p,sx,sy)
        for i in range(sy*my+2,sy*my+my-2):
            for j in range(sx*mx+2,sx*mx+mx-2):
                pixels[j, i] = color

        save_str = str(count).zfill(3)

        i_tmp.save(name + "_"+ save_str + ".png", "PNG")
        count +=1


mx = 10
my = 10
maze, image = generate()
obstacle_list,goal = find_obstacles(maze)
gw = gridworld.Gridworld(10, 0.0, obstacle_list,goal)
createDataSet("1_",gw,maze,image,mx,my)
print(maze)




# # Random Maze Generator using Depth-first Search
# # http://en.wikipedia.org/wiki/Maze_generation_algorithm
# # FB36 - 20130106
# import random
# from PIL import Image
# imgx = 500; imgy = 500
# image = Image.new("RGB", (imgx, imgy))
# pixels = image.load()
# mx = 10; my = 10 # width and height of the maze
# maze = [[0 for x in range(mx)] for y in range(my)]
# dx = [0, 1, 0, -1]; dy = [-1, 0, 1, 0] # 4 directions to move in the maze
# color = [(0, 0, 0), (255, 255, 255), (255,255,0)] # RGB colors of the maze
# # start the maze from a random cell
# cx = random.randint(0, mx - 1); cy = random.randint(0, my - 1)
# maze[cy][cx] = 1; stack = [(cx, cy, 0)] # stack element: (x, y, direction)

# while len(stack) > 0:
#     (cx, cy, cd) = stack[-1]
#     # to prevent zigzags:
#     # if changed direction in the last move then cannot change again
#     if len(stack) > 2:
#         if cd != stack[-2][2]: dirRange = [cd]
#         else: dirRange = range(4)
#     else: dirRange = range(4)

#     # find a new cell to add
#     nlst = [] # list of available neighbors
#     for i in dirRange:
#         nx = cx + dx[i]; ny = cy + dy[i]
#         if nx >= 0 and nx < mx and ny >= 0 and ny < my:
#             if maze[ny][nx] == 0:
#                 ctr = 0 # of occupied neighbors must be 1
#                 for j in range(4):
#                     ex = nx + dx[j]; ey = ny + dy[j]
#                     if ex >= 0 and ex < mx and ey >= 0 and ey < my:
#                         if maze[ey][ex] == 1: ctr += 1
#                 if ctr == 1: nlst.append(i)

#     # if 1 or more neighbors available then randomly select one and move
#     if len(nlst) > 0:
#         ir = nlst[random.randint(0, len(nlst) - 1)]
#         cx += dx[ir]; cy += dy[ir]; maze[cy][cx] = 1
#         stack.append((cx, cy, ir))
#     else: stack.pop()

# # paint the maze
# for ky in range(imgy):
#     for kx in range(imgx):
#         pixels[kx, ky] = color[maze[my * ky // imgy][mx * kx // imgx]]

# ky = random.randint(0,imgy-1)
# kx = random.randint(0,imgx-1)

# while(maze[my * ky // imgy][mx * kx // imgx] != 1):
#     ky = random.randint(0,imgy-1)
#     kx = random.randint(0,imgx-1)

# countup = 0
# countdown = 0
# countleft = 0
# countright = 0

# #up
# kyy=ky-1
# while (kyy > -1 and maze[my * kyy // imgy][mx * kx // imgx] != 0):
#     countup += 1
#     kyy-=1
# #left
# kxx=kx-1
# while (kxx > -1 and maze[my * ky // imgy][mx * kxx // imgx] != 0):
#     countleft += 1
#     kxx-=1
# print("Paint!")
# for i in range(ky-countup,(ky-countup)+imgy//my):
#     for j in range(kx-countleft,(kx-countleft)+imgx//mx):
#         pixels[j, i] = color[2]

# image.save("Maze_" + str(mx) + "x" + str(my) + ".png", "PNG")