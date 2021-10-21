import math
import json

with open("result.json", "r") as json_file:
    keypoint_org = json.load(json_file)

keypoint_org_dict = {}
for frame in keypoint_org:
    image_id = frame["image_id"]
    keypoint_org_dict[int((image_id).split(".")[0].split("frame")[1])] = frame

def getDistance(x, y):
    return math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

def modifyKeypoints(frame_x, frame_y, threshold=2):
    # frame_x is t-1 and fram_y is t
    frame_x_keypoints = frame_x["keypoints"]
    frame_y_keypoints = frame_y["keypoints"]
    return_keypoints = []
    for i in range(0, 51, 3):
        # there are 17 * 3 keypoints
        point_x = frame_x_keypoints[i:i+2]
        point_y = frame_y_keypoints[i:i+2]
        
        if getDistance(point_x, point_y) <= threshold:
            return_keypoints.extend([point_y[0], point_y[1], 1.0])
        else:
            return_keypoints.extend([point_x[0], point_x[1], 1.0])
    return return_keypoints

keypoint_filter_dict = {}

for i in range(len(keypoint_org_dict)):
    if i==0:
        keypoint_filter_dict[i] = keypoint_org_dict[i]["keypoints"]
    else:
        keypoint_filter_dict[i] = modifyKeypoints(keypoint_org_dict[(i-1)], keypoint_org_dict[i])

for i in range(len(keypoint_org_dict)):
    keypoint_org_dict[i]["keypoints"] = keypoint_filter_dict[i]
    
keypoint_filter_list = []

for i in range(len(keypoint_org_dict)):
    keypoint_filter_list.append(keypoint_org_dict[i])

with open("filter_result.json", "w") as json_file:
    json.dump(keypoint_filter_list, json_file)

