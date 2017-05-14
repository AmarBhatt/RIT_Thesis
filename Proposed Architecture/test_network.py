import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
from DataSetGenerator.maze_generator import *
from PIL import Image
import os.path


def test_network(sess, net, X, epoch, num_test, same, location, test_image_location,test_array, normalize, data_size, actual_size,debug=True):
	result = np.zeros(num_test)
	count_max = 100
	out_file = open("results/"+location+"/"+str(epoch)+"_Results.txt",'w')
	total_path = 0
	cur_path_total = 0
	#Test Network
	for t in range(0,num_test):

		image = Image.open(test_image_location+"/"+str(t)+".png")
		state = test_array[t]		
		pixels = list(image.getdata())
		pixels = np.array([pixels[j * actual_size:(j + 1) * actual_size] for j in range(actual_size)],dtype=np.uint8)
		image = Image.fromarray(pixels, 'L')

		f = open("results/"+location+"/"+str(epoch)+"_"+str(t)+".txt",'w')
		data,new_state,gw,failed,done,environment,image = environmentStep(-1,state,100,100,10,10, image = None, gw = None, environment = None,feed=image)
		
		path = gw.bfs(new_state, gw.goal)
		path = path[1:None]
		total_path += len(path)

		#image.show()
		#input("Pause")
		#print(t,failed,done)
		count = 0
		frames = []

		optimal = True
		cur_path = 0
		
		while(not done and not failed):# and count < count_max):
	        #preprocess
			full_image = Image.fromarray(data.squeeze(axis=2), 'L')
			frames.append(full_image)
			#full_image.show()
			#input("Pause")
			img = preprocessing(full_image,data_size)
			pixels = list(img.getdata())
			pixels = np.array([pixels[j * data_size:(j + 1) * data_size] for j in range(data_size)])
			pixels = pixels[:,:,np.newaxis]
			pixels = np.divide(pixels,normalize)
			
			action = sess.run(net,feed_dict={X: [pixels]})
			f.write(str(action))
			f.write('\n')
			action = np.argmax(action[0])
			f.write(str(action))
			f.write('\n')
			#print(action)
			data,new_state,gw,failed,done,environment,image = environmentStep(action,new_state,100,100,10,10,image,gw,environment)

			if(optimal):
				if(path[count]==new_state):
					cur_path += 1
				else:
					optimal = False

			if(count == count_max):
				failed = 1;

			if(failed):
				result[t] = 0
				if(count >= count_max):
					if debug:
						print(str(t)+": You took too long")
					out_file.write(str(t)+": You took too long")
					out_file.write('\n')
				else:
					if debug:
						print(str(t)+": You hit a wall!")
					out_file.write(str(t)+": You hit a wall!")
					out_file.write('\n')
				f.write("FAIL")
				f.write('\n')
			elif(done):
				result[t] = 1
				if debug:
					print(str(t)+": You won!")
				out_file.write(str(t)+": You won!")
				out_file.write('\n')
				f.write("PASS")
				f.write('\n')

			if(failed or done):
				f.write(str(cur_path)+"/"+str(len(path)))
				f.write('\n')
				cur_path_total+=cur_path
			count+=1
		frames[0].save("results/"+location+"/"+str(epoch)+"_"+str(t)+".gif",save_all=True, append_images=frames[1:])
		f.flush()
		f.close()
	#print(len(result))
	result = np.array(result)
	win = np.sum(result)
	lose = result.size-win

	#print(result.size,lose)
	print("After %d tests: %d Passed and %d Failed, Accuracy of: %0.2f, Total Paths Completed: %d out of %d" % (num_test,win,lose,win/num_test,cur_path_total,total_path))
	out_file.write("After %d tests: %d Passed and %d Failed, Accuracy of: %0.2f, Total Paths Completed: %d out of %d" % (num_test,win,lose,win/num_test,cur_path_total,total_path))
	out_file.write('\n')
	out_file.flush()
	out_file.close()

	return win,lose,cur_path_total,total_path