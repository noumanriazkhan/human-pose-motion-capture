import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

with open("human_detection.json", "r") as f:
    human_dections = json.load(f)


def draw_bboxes(bboxes, image):
    for bbox in bboxes:
        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    return image


def change_bbox_format(bboxes):
    changed_bboxes = []
    for bbox in bboxes:
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(x1 + bbox[2])
        y2 = int(y1 + bbox[3])
        changed_bboxes.append([x1, y1, x2, y2])
    return changed_bboxes


def adjust_bbox(bboxes):
    # bboxes = change_bbox_format(bboxes)
    bboxes = np.asarray(bboxes)
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    new_bbox = [min(x1), min(y1), max(x2), max(y2)]
    return new_bbox


def change_bbox_hw(bboxes):
    changed_bboxes = []
    for bbox in bboxes:
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2] - x1)
        y2 = int(bbox[3] - y1)
        changed_bboxes.append([x1, y1, x2, y2])
    return changed_bboxes


def cal_avg_bbox(bboxes):
    changed_bbox = bboxes.copy()
    for i in range(15, len(bboxes) - 15):
        sub_bboxes = bboxes[i-15:i+15]
        if i == 145:
            print(sub_bboxes)
        adj_bbox = adjust_bbox(sub_bboxes)
        changed_bbox[i] = adj_bbox
    return changed_bbox
    

def collect_bbox(human_detections):
    bboxes = []
    for detection in human_dections:
        bboxes.append(detection["bbox"])
    return bboxes
    


bboxes = collect_bbox(human_detections=human_dections)
bboxes = change_bbox_format(bboxes=bboxes)
sub_bboxes = change_bbox_hw(cal_avg_bbox(bboxes=bboxes))

for i in range(len(human_dections)):
    human_dections[i]["bbox"] = sub_bboxes[i]

with open("human_detection.json", "w") as f:
    json.dump(human_dections, fp=f)