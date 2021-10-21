import cv2
import numpy as np
import json
import glob
import argparse
import os
from enum import Enum

class CocoBaseline(Enum):
  nose = 0
  left_eye = 1
  right_eye = 2
  left_ear = 3
  right_ear = 4
  left_shoulder = 5
  right_shoulder = 6
  left_elbow = 7
  right_elbow = 8
  left_wrist = 9
  right_wrist = 10
  left_hip = 11
  right_hip = 12
  left_knee = 13
  right_knee = 14
  left_ankle = 15
  right_ankle = 16

CocoBasePairs = [
                  [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
                  [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
                  ]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 85]]

def draw_humans(npimg, json, imgcopy=False):
  if imgcopy:
      npimg = np.copy(npimg)
  image_h, image_w = npimg.shape[:2]
  centers = {}
  for human in json:
      # draw point
      poseKPs = np.array(human['keypoints']).reshape(-1,3)
      for i, poseKP in enumerate(poseKPs):
          center = (int(poseKP[0]), int(poseKP[1]))
          centers[i+1] = center
          cv2.circle(npimg, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)
      # draw line
      for pair_order, pair in enumerate(CocoBasePairs):
          if pair[0] not in range(len(human['keypoints'])) or pair[1] not in range(len(human['keypoints'])):
              continue
          cv2.line(npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)
  return npimg

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Taking arguments to JSON and Images")
	parser.add_argument('--img_folder', help="The path to input images")
	parser.add_argument('--json_path', help="Path to JSON Files")
	args = parser.parse_args()

	resultsDir = os.path.join(args.img_folder,'results')
	if not os.path.exists(resultsDir):
		os.makedirs(os.path.join(resultsDir, 'orig'))
		os.makedirs(os.path.join(resultsDir, 'blank'))

	json_file = json.load( open(args.json_path, 'r') )
	
	img_list = np.unique([w['image_id'] for w in json_file])
	
	for i in range(2):
		for imgID in img_list:
			print('Processing Image ID: {}'.format(imgID))
			kpData = [w for w in json_file if w['image_id'] == imgID]
			img = cv2.imread(os.path.join(args.img_folder, imgID))
			
			if i == 0:
				bimg = np.zeros( (img.shape) )
				img_with_pose = draw_humans(bimg, kpData)
				cv2.imwrite(os.path.join(resultsDir, 'blank', 'pred_'+imgID), img_with_pose.astype(np.uint8))

			else:
				img_with_pose = draw_humans(img, kpData)
				cv2.imwrite(os.path.join(resultsDir, 'orig', 'pred_'+imgID), img_with_pose.astype(np.uint8))