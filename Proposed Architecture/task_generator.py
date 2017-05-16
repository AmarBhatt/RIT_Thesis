# Random Maze Generator using Depth-first Search
# http://en.wikipedia.org/wiki/Maze_generation_algorithm
# FB - 20121214
import random
from PIL import Image
import grid_world as gridworld
import sys as sys
import numpy as np
from matplotlib import pyplot as plt
#import gifmaker

def generate(imgx,imgy,mx,my,num_goals,image=None):
    #imgx = 100; imgy = 100
    maze = [[0 for x in range(mx)] for y in range(my)]
    dx = [0, 1, 0, -1]; dy = [-1, 0, 1, 0] # 4 directions to move in the maze
    #color = [(0,0, 0), (255, 255, 255), (255,255,0)] # RGB colors of the maze
    
    if num_goals > 3:
        print("Too many goals defined.")
        sys.exit()

    goal_colors = [255,0,32,128] #first one is free space!

    #color = [0,255,128]
    if image == None: #create new image
        image = Image.new("L", (imgx, imgy))
        pixels = image.load()
        #mx = 10; my = 10 # width and height of the maze
                
        # get random placement of goals
        goal_placement = np.random.choice(range(0,mx*my),replace=False,size=num_goals)


        for goal in len(goal_placement):
            sx = goal_placement[goal] % mx
            sy = goal_placement[goal] // m

            maze[sy][sx] = goal+1


        # paint the maze
        for ky in range(imgy):
            for kx in range(imgx):
                pixels[kx, ky] = goal_colors[maze[my * ky // imgy][mx * kx // imgx]]
        
    else:
        image = image.copy()
        pixels = image.load()
        for ky in range(imgy):
            for kx in range(imgx):
                if(pixels[kx,ky] == 0):
                    maze[ky//my][kx//mx] = 1
                elif(pixels[kx,ky] == 32):
                    maze[ky//my][kx//mx] = 2
                elif(pixels[kx,ky] == 128):
                    maze[ky//my][kx//mx] = 3

    image.save("Task_" + str(mx) + "x" + str(my) + ".png", "PNG")
    
    return maze, image

def preprocessing(img,size):
    basewidth = size
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.LANCZOS)
    return img

def find_obstacles(maze):
    obstacles = []
    goal = 0
    count = 0
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if(maze[y][x] == -1):
                obstacles.append(count)
            if(maze[y][x] > 0):
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



def createDataSet(name,tw,maze,image, mx, my, imgx,imgy):

    start = random.randint(0,tw.grid_size**2 - 1)
    sx,sy = tw.int_to_point(start)
    frames = []
    while(tw.world_grid[sy][sx] > 0):
        start = random.randint(0,tw.grid_size**2 - 1)
        sx,sy = tw.int_to_point(start)
    path = tw.bfs(start,tw.goals[0])
    count = 0
    color = 64

    f = open(name+".txt",'w')

    act_mx = imgx//mx
    act_my = imgy//my


    for goal in len(tw.goals):
        path = tw.bfs(start,tw.goals[goal])
        count = 0





    for p in range(len(path)): #DOES NOT HIT GOAL, but NEXT ACTION REFLECTS GOAL
        i_tmp = image.copy()
        pixels = i_tmp.load()
        sx,sy = gw.int_to_point(path[p])

        #print(p,sx,sy)
        for i in range(sy*act_my+2,sy*act_my+act_my-2):
            for j in range(sx*act_mx+2,sx*act_mx+act_mx-2):
                pixels[j, i] = color
        if(p == len(path)-1):#at goal so stay there
            action = 4
        else:
            action = gw.optimal_policy(path[p])
        f.write(str(action))
        f.write('\n')
        frames.append(i_tmp)
    frames[0].save(name+".gif", save_all=True, append_images=frames[1:])
    f.flush()
    f.close()
    #count +=1

def processGIF(infile,size):
    try:
        im = Image.open(infile+".gif")
    except IOError:
        print ("Cant load", infile)
        sys.exit(1)
    i = 0
    mypalette = im.getpalette()

    with open(infile+".txt") as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
    actions = [int(x.strip()) for x in content]
    num = len(actions)

    store_image = np.zeros(shape=[num,size,size,1], dtype=np.uint8)
    store_action = np.zeros(shape=[num,5], dtype=np.uint8)
    try:
        while 1:
            im.putpalette(mypalette)
            new_im = Image.new("L", im.size)
            new_im.paste(im)
            #new_im.show()
            new_im = preprocessing(new_im,size)
            #new_im.save('foo'+str(i)+'.png')
            pixels = list(new_im.getdata())
            pixels = np.array([pixels[j * size:(j + 1) * size] for j in range(size)])
            pixels = pixels[:,:,np.newaxis]
            #print(pixels.shape)
            #print(store_image.shape)
            #print(i)
            store_image[i,:,:,:] = pixels#new_im.load()
            action = np.zeros(shape=[5], dtype=np.uint8)
            action[actions[i]] = 1
            store_action[i] = action
            i += 1
            im.seek(im.tell() + 1)

    except EOFError:
        pass # end of sequence

    return store_image, store_action


def testDataAquisition(fname):
    image_set,action_set = processGIF('expert_data/random/'+fname)
    #print(image_set)
    print(action_set)
    plt.gray()
    for i in image_set:
        #plt.imshow(image_set[i,:,:,:])
        #plt.show()
        img = Image.fromarray(i.squeeze(axis=2), 'L')
        img.show()
        input("Press Enter to continue...")


def environmentStep(action,state,imgx,imgy,mx,my, image = None, gw = None, environment = None,feed=None):
    color = 64
    failed = False
    done = False
    if action == -1: #create random enviroment
        maze, image = generate(imgx,imgy,mx,my,image=feed)
        obstacle_list,goal = find_obstacles(maze)
        gw = gridworld.Gridworld(mx, 0.0, obstacle_list,goal)
        
        if(state > -1):
            start = state
            sx,sy = gw.int_to_point(start)
        else:
            start = random.randint(0,gw.grid_size**2 - 1)
            sx,sy = gw.int_to_point(start)
            while(gw.world_grid[sy][sx] > 0):
                start = random.randint(0,gw.grid_size**2 - 1)
                sx,sy = gw.int_to_point(start)
        i_tmp = image.copy()

        pixels = i_tmp.load()
        #print(p,sx,sy)
        for i in range(sy*my+2,sy*my+my-2):
            for j in range(sx*mx+2,sx*mx+mx-2):
                pixels[j, i] = color

        data = list(i_tmp.getdata())
        data = np.array([data[k * imgx:(k + 1) * imgx] for k in range(imgx)], dtype=np.uint8)
        data = data[:,:,np.newaxis]
        
        environment = list(image.getdata())
        environment = np.array([environment[r * imgx:(r + 1) * imgx] for r in range(imgx)], dtype=np.uint8)

        new_state = start
    else:
        if(action == 0): #go right
            sx,sy = gw.int_to_point(state)
            sx += 1
        if(action == 1): #go down
            sx,sy = gw.int_to_point(state)
            sy += 1
        if(action == 2): #go left
            sx,sy = gw.int_to_point(state)
            sx -= 1
        if(action == 3): #go up
            sx,sy = gw.int_to_point(state)
            sy -= 1
        if(action == 4): #stay
            sx,sy = gw.int_to_point(state)

        new_state = gw.point_to_int([sx,sy])
        if (sx < 0 or sx == mx or sy < 0 or sy == my):
            failed = 1
        else:
            failed = gw.world_grid[sy][sx] == 1
            done = gw.world_grid[sy][sx] == 2
        i_tmp = image.copy()
        pixels = i_tmp.load()
        #print(p,sx,sy)
        if(not failed):
            for i in range(sy*my+2,sy*my+my-2):
                for j in range(sx*mx+2,sx*mx+mx-2):
                    pixels[j, i] = color
        data = list(i_tmp.getdata())
        data = np.array([data[j * imgx:(j + 1) * imgx] for j in range(imgx)], dtype=np.uint8)
        data = data[:,:,np.newaxis]


    return data,new_state,gw,failed,done,environment,image

def environmentStepTest(imgx,imgy,mx,my):
    data,new_state,gw,failed,done,environment,image = environmentStep(-1,-1,imgx,imgy,mx,my, image = None, gw = None, environment = None)
    img = Image.fromarray(data.squeeze(axis=2), 'L')
    img.show()
    while(not done and not failed):
        x = int(input("Enter action (0-4):"))
        data,new_state,gw,failed,done,environment,image = environmentStep(x,new_state,imgx,imgy,mx,my,image,gw,environment)
        print(done,failed)
        img = Image.fromarray(data.squeeze(axis=2), 'L')
        img.show()
    if(failed):
        print("You hit a wall!")
    if(done):
        print("You won!")


def createRandomDataSet(name,gw,maze,image, mx, my, imgx,imgy):

    start = random.randint(0,gw.grid_size**2 - 1)
    sx,sy = gw.int_to_point(start)
    frames = []
    while(gw.world_grid[sy][sx] > 0):
        start = random.randint(0,gw.grid_size**2 - 1)
        sx,sy = gw.int_to_point(start)
    path = gw.bfs(start,gw.goal)
    count = 0
    color = 64

    f = open(name+".txt",'w')

    act_mx = imgx//mx
    act_my = imgy//my

    #print(gw.world_grid)
    #print(sx,sy)
    #input("Pause")

    for p in range(0,2):
        valid = False
        i_tmp = image.copy()
        pixels = i_tmp.load()
        #print(p,sx,sy)
        for i in range(sy*act_my+2,sy*act_my+act_my-2):
            for j in range(sx*act_mx+2,sx*act_mx+act_mx-2):
                pixels[j, i] = color
        action = np.random.choice(5, size=1)[0]
        while not valid:        
            if action == 0 and sx+1 < act_mx:
                if gw.world_grid[sy,sx+1] == 0 or gw.world_grid[sy,sx+1] == 2:
                    val_action = 0
                    sy = sy
                    sx = sx+1
                    valid = True
            elif action == 2 and sx-1 >= 0:
                if gw.world_grid[sy,sx-1] == 0 or gw.world_grid[sy,sx-1] == 2:
                    val_action = 2
                    valid = True
                    sy = sy
                    sx = sx-1
            elif action == 1 and sy+1 < act_my:
                if gw.world_grid[sy+1,sx] == 0 or gw.world_grid[sy+1,sx] == 2:
                    val_action = 1
                    valid = True
                    sy = sy+1
                    sx = sx    
            elif action == 3 and sy-1 >= 0:
                if gw.world_grid[sy-1,sx] == 0 or gw.world_grid[sy-1,sx] == 2:
                    val_action = 3
                    valid = True 
                    sy = sy-1
                    sx = sx   
            elif action == 4:# or gw.world_grid[sy,sx] == 2:
                val_action = 4
                valid = True 
                sy = sy
                sx = sx   
            #print(sy,sx,action)
            #input("Pause")
            action = np.random.choice(5, size=1)[0]
            #print(action)

        f.write(str(val_action))
        f.write('\n')
        frames.append(i_tmp)
    frames[0].save(name+".gif", save_all=True, append_images=frames[1:])
    f.flush()
    f.close()


def bulkCreate(imgx,imgy,mx,my,num_goals,num_gen,num_gen_rew,num_gen_test,location,rew_location,test_location,rmaze):
    
    if rmaze:
        for i in range(num_gen):
            print(i)
            maze, image = generate(imgx,imgy,mx,my,num_goals)
            obstacle_list,goal = find_obstacles(maze)
            tw = taskworld.Taskworld(mx, 0.0, goal)
            file = location+"/"+str(i)
            createDataSet(file,tw,maze,image,mx,my,imgx,imgy)
        for i in range(num_gen_rew):
            print(i)
            maze, image = generate(imgx,imgy,mx,my)
            obstacle_list,goal = find_obstacles(maze)
            gw = gridworld.Gridworld(mx, 0.0, goal)
            file = rew_location+"/"+str(i)
            createRandomDataSet(file,gw,maze,image,mx,my,imgx,imgy)
        for i in range(num_gen_test):
            print(i)
            file = test_location+"/"+str(i)
            maze, image = generate(imgx,imgy,mx,my)
            im = image.copy()
            pixels = list(im.getdata())
            pixels = np.array([pixels[j * imgx:(j + 1) * imgx] for j in range(imgx)])
            for i in range(imgx):
                for j in range(imgx):
                    if(pixels[j,i] == 64):
                        pixels[j, i] = 255
            im.save(file+ ".png", "PNG")
    else: #use the same map
        maze, image = generate(imgx,imgy,mx,my)
        obstacle_list,goal = find_obstacles(maze)
        gw = gridworld.Gridworld(mx, 0.0, obstacle_list,goal)
        for i in range(num_gen):
            print(i)
            file = location+"/"+str(i)
            createDataSet(file,gw,maze,image,mx,my,imgx,imgy)
        for i in range(num_gen_rew):
            print(i)
            file = rew_location+"/"+str(i)
            createRandomDataSet(file,gw,maze,image,mx,my,imgx,imgy)
        for i in range(num_gen_test):
            print(i)
            file = test_location+"/"+str(i)
            im = image.copy()
            pixels = list(im.getdata())
            pixels = np.array([pixels[j * imgx:(j + 1) * imgx] for j in range(imgx)])
            for i in range(imgx):
                for j in range(imgx):
                    if(pixels[j,i] == 64):
                        pixels[j, i] = 255
            im.save(file+ ".png", "PNG")
        #print(maze)
        #processGIF('test.gif')


def pix2img(pixels,show=False):
    pixels = np.array(pixels,dtype=np.uint8)
    image = Image.fromarray(pixels,'L')
    if show:
        image.show()
    return image


imgx = 100 #100
imgy = 100 #100
mx = 10 #16 #10
my = 10 #16 #10
#bulkCreate(imgx,imgy,mx,my,10000,10000,10,"expert_data/random","random_data/random","test_data/random",True)
#processGIF("expert_data/random/0",10)
#maze, image = generate(imgx,imgy,mx,my)
#environmentStepTest(imgx,imgy,mx,my)