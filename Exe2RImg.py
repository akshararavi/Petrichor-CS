# Binary to Image Converter
# Read executable binary files and convert them RGB and greyscale png images

import os, math
from PIL import Image
from math import sqrt, ceil
import numpy as np
from pathlib import Path

#Input file name
file_path = ['./Small_Dataset/train/Benignware', './Small_Dataset/train/Malware', './Small_Dataset/test/Benignware', './Small_Dataset/test/Malware']
cimage_path = ['./RGB_Data224/train/Benignware', './RGB_Data224/train/Malware', './RGB_Data224/test/Benignware', './RGB_Data224/test/Malware']
IMG_SIZE = 224

def getBinaryData(filename):
	"""
	Extract byte values from binary executable file and store them into list
	:param filename: executable file name
	:return: byte value list
	"""
	binary_values = []

	#with open(os.path.join(file_path, filename), 'rb') as fileobject:
	with open(filename, 'rb') as fileobject:

		# read file byte by byte
		data = fileobject.read(1)

		while data != b'':
			binary_values.append(ord(data))
			data = fileobject.read(1)

	return binary_values

def createRGBImage(filename, image_path):
	"""
	Create RGB image from 24 bit binary data 8bit Red, 8 bit Green, 8bit Blue
	:param filename: image filename
	"""
	index = 0
	rgb_data = []

	# Read binary file
	binary_data = getBinaryData(filename)

	# Create R,G,B pixels
	while (index + 3) <= len(binary_data):
		R = binary_data[index]
		G = binary_data[index+1]
		B = binary_data[index+2]
		index += 3
		rgb_data.append((R, G, B))

	remain_bytes = len(binary_data) - (3*len(rgb_data))
	if(remain_bytes == 1):
		R = binary_data[(3*len(rgb_data))]
		G = 0
		B = 0
		rgb_data.append((R, G, B))
	elif(remain_bytes == 2):
		R = binary_data[(3*len(rgb_data))]
		G = binary_data[(3*len(rgb_data))+1]
		B = 0
		rgb_data.append((R, G, B))
		
	file_len = IMG_SIZE * IMG_SIZE
	data_length = len(rgb_data)
	pad_len   = max((file_len - data_length), 0)
	print(f"rgb_len: {data_length}")
	print(f"pad_len: {pad_len}")
	
	# Pad data with zeros at the end.
	for i in range(pad_len):
		rgb_data.append((0, 0, 0))

	save_file(filename, image_path, rgb_data[0:file_len], (IMG_SIZE, IMG_SIZE), 'RGB')


def save_file(filename, image_path, data, size, image_type):
    '''
	if image_type == 'L':
		image_path = gimage_path
	else:
		image_path = cimage_path
    '''
    try:
        # Save image
        image = Image.new(image_type, size)
        image.putdata(data)
        #print(filename)
        #filename = Path(filename).stem
        filename = os.path.split(filename)[-1]
        print(filename)
        imagename = image_path + os.sep + filename + '.png'
        print(imagename)
        image.save(imagename)
    except Exception as err:
        print(err)

if __name__ == '__main__':

    for i in range(len(file_path)):
	    count = 1
	    for filename in os.listdir(file_path[i]):
	        print(count)
	        filename = os.path.join(file_path[i], filename)
	        print(filename)
	        createRGBImage(filename, cimage_path[i])
	        count = count + 1
