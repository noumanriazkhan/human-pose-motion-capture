"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np
import tqdm
import skimage.io as io
import tensorflow as tf
import json
from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

import pandas as pd 
import os
import glob
import cv2

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')



def get_bbox(js):
  kps = np.array(js['keypoints']).reshape(-1,3)[:,:2]
  min_pt = np.min(kps, axis=0)
  max_pt = np.max(kps, axis=0)
  person_height = np.linalg.norm(max_pt - min_pt)
  center = (min_pt + max_pt) / 2.
  scale = 150. / person_height

  return scale, center

def resize_img(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor


def scale_and_crop(image, scale, center, img_size):
    image_scaled, scale_factors = resize_img(image, scale)
    # Swap so it's [x, y]
    scale_factors = [scale_factors[1], scale_factors[0]]
    center_scaled = np.round(center * scale_factors).astype(np.int)

    margin = int(img_size / 2)
    image_pad = np.pad(
        image_scaled, ((margin, ), (margin, ), (0, )), mode='edge')
    center_pad = center_scaled + margin
    # figure out starting point
    start_pt = center_pad - margin
    end_pt = center_pad + margin
    # crop:
    crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    proc_param = {
        'scale': scale,
        'start_pt': start_pt,
        'end_pt': end_pt,
        'img_size': img_size
    }

    return crop, proc_param

def pixel_to_point(pixel, depth):
	ppx = 320.9
	ppy = 176.1
	fx = 322.3
	fy = 322.3

	ax = (pixel[0] - ppx) / fx
	ay = (pixel[1] - ppy) / fy

	X =  depth * ay
	Y = depth * ax
	Z = depth

	return [Y, X, Z]

def to36M(bones, body_parts):
    H36M_JOINTS_17 = [
        'Hip', 
        'RHip',
        'RKnee',
        'RFoot',
        'LHip',
        'LKnee',
        'LFoot',
        'Spine',
        'Thorax',
        'Neck/Nose',
        'Head',
        'LShoulder',
        'LElbow',
        'LWrist',
        'RShoulder',
        'RElbow',
        'RWrist',
    ]
    adjusted_bones = []
    for name in H36M_JOINTS_17:
        if not name in body_parts:
            if name == 'Hip':
                adjusted_bones.append((bones[body_parts['RHip']] + bones[body_parts['LHip']]) / 2)
            elif name == 'RFoot':
                adjusted_bones.append(bones[body_parts['RAnkle']])
            elif name == 'LFoot':
                adjusted_bones.append(bones[body_parts['LAnkle']])
            elif name == 'Spine':
                adjusted_bones.append(
                    (
                            bones[body_parts['RHip']] + bones[body_parts['LHip']]
                            + bones[body_parts['RShoulder']] + bones[body_parts['LShoulder']]
                    ) / 4
                )
            elif name == 'Thorax':
                adjusted_bones.append(
                    (
                            + bones[body_parts['RShoulder']] + bones[body_parts['LShoulder']]
                    ) / 2
                )
            elif name == 'Head':
                thorax = (
                                 + bones[body_parts['RShoulder']] + bones[body_parts['LShoulder']]
                         ) / 2
                adjusted_bones.append(
                    thorax + (
                            bones[body_parts['Nose']] - thorax
                    ) * 2
                )
            elif name == 'Neck/Nose':
                adjusted_bones.append(bones[body_parts['Nose']])
            else:
                raise Exception(name)
        else:
            adjusted_bones.append(bones[body_parts[name]])
    return adjusted_bones


def parts(dataset='COCO'):
    if dataset == 'COCO':
        BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                      "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}
        POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                      ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                      ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                      ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
    else:
        assert (dataset == 'MPI')
        BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                      "Background": 15}
        POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                      ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                      ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]
    return BODY_PARTS, POSE_PAIRS

def preprocess_image(img_path, js=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if js is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = get_bbox(js)

    crop, proc_param = scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main(img_path, json_path=None):

    joints_names = ['Ankle.R_x', 'Ankle.R_y', 'Ankle.R_z',
                   'Knee.R_x', 'Knee.R_y', 'Knee.R_z',
                   'Hip.R_x', 'Hip.R_y', 'Hip.R_z',
                   'Hip.L_x', 'Hip.L_y', 'Hip.L_z',
                   'Knee.L_x', 'Knee.L_y', 'Knee.L_z', 
                   'Ankle.L_x', 'Ankle.L_y', 'Ankle.L_z',
                   'Wrist.R_x', 'Wrist.R_y', 'Wrist.R_z', 
                   'Elbow.R_x', 'Elbow.R_y', 'Elbow.R_z', 
                   'Shoulder.R_x', 'Shoulder.R_y', 'Shoulder.R_z', 
                   'Shoulder.L_x', 'Shoulder.L_y', 'Shoulder.L_z',
                   'Elbow.L_x', 'Elbow.L_y', 'Elbow.L_z',
                   'Wrist.L_x', 'Wrist.L_y', 'Wrist.L_z', 
                   'Neck_x', 'Neck_y', 'Neck_z', 
                   'Head_x', 'Head_y', 'Head_z', 
                   'Nose_x', 'Nose_y', 'Nose_z', 
                   'Eye.L_x', 'Eye.L_y', 'Eye.L_z', 
                   'Eye.R_x', 'Eye.R_y', 'Eye.R_z', 
                   'Ear.L_x', 'Ear.L_y', 'Ear.L_z', 
                   'Ear.R_x', 'Ear.R_y', 'Ear.R_z']
  
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    json_file = json.load( open(json_path, 'r') )

    for js in json_file:

      imgID = js['image_id']
      kps = np.array(js['keypoints']).reshape(-1,3)[:,:2]
      points = np.vstack( (kps[0], (kps[5] + kps[6])/2, kps[6], kps[8], kps[10], kps[5], kps[7], kps[9], kps[12], kps[14], kps[16], kps[11], kps[13], kps[15], kps[2], kps[1], kps[4], kps[3], [0.0, 0.0] ))
      points = [vec for vec in points]
      points = [np.array(vec) for vec in points]
      BODY_PARTS, POSE_PAIRS = parts()
      points = to36M(points, BODY_PARTS)
      pose_proj = pixel_to_point(points[0], 0.8)

      input_img, proc_param, img = preprocess_image(os.path.join(img_path,imgID))

      input_img = np.expand_dims(input_img, 0)

      joints, verts, cams, joints3d, theta = model.predict(input_img, get_theta=True)
      j3d = joints3d[0].reshape(1,-1)

      j3d.T[0::3] += pose_proj[0]
      j3d.T[1::3] += pose_proj[1]
      j3d.T[2::3] += pose_proj[2]

      joints_export = pd.DataFrame(j3d.reshape(1,57), columns=joints_names)

      joints_export.index.name = 'frame'
      joints_export.iloc[:, 1::3] = joints_export.iloc[:, 1::3]*-1
      joints_export.iloc[:, 2::3] = joints_export.iloc[:, 2::3]*-1

  #     col_list = list(joints_export)

  #     col_list[1::3], col_list[2::3] = col_list[2::3], col_list[1::3]

  #     joints_export = joints_export[col_list]
      hipCenter = joints_export.loc[:][['Hip.R_x', 'Hip.R_y', 'Hip.R_z',
                                        'Hip.L_x', 'Hip.L_y', 'Hip.L_z']]

      joints_export['hip.Center_x'] = hipCenter.iloc[0][::3].sum()/2
      joints_export['hip.Center_y'] = hipCenter.iloc[0][1::3].sum()/2
      joints_export['hip.Center_z'] = hipCenter.iloc[0][2::3].sum()/2

      joints_export.to_csv("./output/csv/"+imgID.replace('jpg','csv'))

def join_csv():
  path = './output/csv/'                   
  all_files = glob.glob(os.path.join(path, "*.csv")    )

  df_from_each_file = (pd.read_csv(f) for f in sorted(all_files))
  concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
  concatenated_df.drop('frame', axis=1, inplace=True)
  concatenated_df_mean = concatenated_df.rolling(5).sum()
  final_df = concatenated_df_mean.iloc[4:,:]
  final_df.index = np.arange(1, len(final_df) + 1)
  final_df.to_csv("./output/csv_joined/csv_joined.csv", index=True, index_label='frame')

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main('../../inzamam/videoframes/hmr/', '../../inzamam/result.json')
    
    join_csv()
    
    print('\nResult is in hmr/output (you can open images in Colaboratory by double-clicking them)')

