import cv2
import json
import tqdm

json_file = json.load( open( 'human_detection.json', 'r') )

for js in tqdm.tqdm(json_file):
	img = cv2.imread('./videoframes/'+js['image_id'])
	bbox = [int(w) for w in js['bbox']]
	ximg = img[ bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2] ]
	ximg = cv2.resize( ximg, (ximg.shape[1], 150) )
	cv2.imwrite('./videoframes/hmr/'+js['image_id'], ximg)
