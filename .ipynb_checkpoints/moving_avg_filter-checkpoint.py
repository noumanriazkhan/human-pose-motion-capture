import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

window_size = 10

with open("result.json", "r") as json_file:
    keypoints = json.load(json_file)

keypoints_list = []
for i in range(len(keypoints)):
    keypoints_list.append(keypoints[i]["keypoints"])

# value at index zero represent frame
keypoints_list.reverse()

np_keypoints = np.asarray(keypoints_list)
pd_keypoints = pd.DataFrame(data=np_keypoints)

pd_keypoints_mean = pd_keypoints.copy()

for i in range(51):
    pd_keypoints_mean[i] = pd.Series(pd_keypoints[i]).rolling(window_size).mean()

# first number of frames equal to windows_size are NaN
# so leaving thoes frames as it is
for i in range(len(keypoints)):
    if i >= (window_size - 1):
        keypoints[len(keypoints) - 1 - i]["keypoints"] = pd_keypoints_mean.iloc[i].values.tolist()

with open("filtered_result.json", "w") as json_file:
    json.dump(keypoints, fp=json_file)

