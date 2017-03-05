import scipy as sci
import scipy.ndimage
import numpy as np

# Read Image

def readImage(file):
	image = sci.ndimage.imread(file, True)
	#print(image)
	return image