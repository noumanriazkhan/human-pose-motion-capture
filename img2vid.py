import cv2
import numpy as np
import glob
import argparse
import os

def write_to_video(path_to_folder):
	sub_folds = ['orig', 'blank']

	for sf in sub_folds:
		img_array = []
		for filename in sorted(glob.glob(os.path.join(path_to_folder, sf, '*.jpg'))):
			img = cv2.imread(filename)
			height, width, layers = img.shape
			size = (width,height)
			img_array.append(img)

		out = cv2.VideoWriter(os.path.join(path_to_folder, 'output_'+sf+'.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
		
		for i in range(len(img_array)):
			out.write(img_array[i])
		out.release()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Taking arguments to write videos")
	parser.add_argument('--img_folder', help="Folder with image subfolders")
	args = parser.parse_args()

	write_to_video(args.img_folder)
