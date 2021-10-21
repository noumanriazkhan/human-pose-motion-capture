import chainer
import cv2 as cv
import numpy as np
import argparse
import pandas as pd
import sys
import os
import json
import evaluation_util
import tqdm
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import projection_gan

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


def parts(args):
    if args.dataset == 'COCO':
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
        assert (args.dataset == 'MPI')
        BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                      "Background": 15}
        POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                      ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                      ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]
    return BODY_PARTS, POSE_PAIRS


class OpenPose(object):
    """
    This implementation is based on https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py
    """

    def __init__(self, args):
        self.net = cv.dnn.readNetFromCaffe(args.proto2d, args.model2d)
        if args.inf_engine:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)

    def predict(self, args, frame):

        inWidth = args.width
        inHeight = args.height

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        inp = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                   (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inp)
        out = self.net.forward()

        BODY_PARTS, POSE_PAIRS = parts(args)

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            points.append((x, y) if conf > args.thr else None)
        return points


def create_pose(model, points):
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        x = points[:, 0::2]
        y = points[:, 1::2]
        z_pred = model(points).data

        pose = np.stack((x, y, z_pred), axis=-1)
        pose = np.reshape(pose, (len(points), -1))

        return pose

def write_to_csv(points, args, idx):
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
                   'hip.Center_x','hip.Center_y','hip.Center_z']

    p = points.reshape(-1,3)
    final_points = np.array([ p[3], p[2], p[1], p[4], p[5], p[6], p[16], p[15], p[14], p[11], p[12], p[13], p[8], p[10], p[9], p[0] ]).reshape(1,-1)
    joints_export = pd.DataFrame(final_points, columns=joints_names)
    joints_export.index.name = 'frame'
    joints_export.iloc[:, 1::3] = joints_export.iloc[:, 1::3]*-1
    joints_export.iloc[:, 2::3] = joints_export.iloc[:, 2::3]*-1

    joints_export.to_csv('./csv_out/' + args.input.split('/')[-1].split('.')[0] + str(idx) + ".csv")

def join_csv():
    path = './csv_out/'                   
    all_files = glob.glob(os.path.join(path, "*.csv"))

    df_from_each_file = (pd.read_csv(f) for f in sorted(all_files))
    concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)

    #concatenated_df['frame'] = concatenated_df.index+1
    concatenated_df.drop('frame', axis=1, inplace=True)
    concatenated_df_mean = concatenated_df.rolling(5).sum()
    final_df = concatenated_df_mean.iloc[4:,:]
    final_df.index = np.arange(1, len(final_df) + 1)
    final_df.to_csv(path + "csv_joined.csv", index=True, index_label='frame')

def normalize_2d(pose):
    xs = pose.T[0::2] - pose.T[0]
    ys = pose.T[1::2] - pose.T[1]
    scale = np.sqrt(xs[1:] ** 2 + ys[1:] ** 2).mean(axis=0)
    pose = pose.T / scale
    mu_x = pose[0].copy()
    mu_y = pose[1].copy()
    pose[0::2] -= mu_x
    pose[1::2] -= mu_y
    return mu_x, mu_y, scale, pose.T

def normalize_2d_with_xy(pose):
    mu_x = pose.T[0].copy()
    mu_y = pose.T[1].copy()
    xs = pose.T[0::2] - pose.T[0]
    ys = pose.T[1::2] - pose.T[1]
    scale = np.sqrt(xs[1:] ** 2 + ys[1:] ** 2).mean(axis=0)
    pose = pose.T / scale
    pose[0::2] -= pose[0].copy()
    pose[1::2] -= pose[1].copy()
    return mu_x, mu_y, scale, pose.T

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

def angle_between_3dpoints(a,b):
	ab = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
	aMode = np.sqrt( a[0]*a[0] + a[1]*a[1] + a[2]*a[2] )
	bMode = np.sqrt( b[0]*b[0] + b[1]*b[1] + b[2]*b[2] )
	return np.arccos(ab / (aMode*bMode))

def direction_angles(a):
	aMode = np.linalg.norm(np.array(a))
	alpha = a[0]/aMode
	beta = a[1]/aMode
	gamma = a[2]/aMode
	return [alpha, beta, gamma]

def main(args):
    model = evaluation_util.load_model(vars(args))
    chainer.serializers.load_npz(args.lift_model, model)

    if args.mode == 'video':
        cap = cv.VideoCapture(args.input if args.input else 0)

        idx = 0

        while(cap.isOpened()):
            hasFrame, frame = cap.read()
            print(idx)
            points = OpenPose(args).predict(args, frame)
            points = [vec for vec in points]
            points = [np.array(vec) for vec in points]
            if [p for p in points if np.any(p == None)]:
                continue
            BODY_PARTS, POSE_PAIRS = parts(args)
            points = to36M(points, BODY_PARTS)
            points = np.reshape(points, [1, -1]).astype('f')
            #points_norm = projection_gan.pose.dataset.pose_dataset.pose_dataset_base.Normalization.normalize_2d(points)
            mu_x, mu_y, points_norm = normalize_2d(points)
            pose = create_pose(model, points_norm)
            write_to_csv(pose, args, idx)
            #out_img = evaluation_util.create_img(points[0], frame)
            #cv.imwrite('./csv_out/openpose_detect{}.jpg'.format(idx), out_img)
            idx += 1
    else:
        assert(args.mode == 'json')
        json_file = json.load( open(args.input, 'r') )

        prev_mu_x, prev_mu_y = 0.0,0.0
        first_frame = True
        prev_hip_loc = [0,0,0]
        #depth_df = pd.read_csv('./result_dense.csv')
        #depth_js = json.load( open('depth_map.json','r') )

        for js in tqdm.tqdm(reversed(json_file)):
            '''
            path = os.path.join(args.img_folder, imgID)
            if not os.path.isfile(path):
                raise Exception('Invalid Image Path')
            img = cv.imread(path)
            points = OpenPose(args).predict(args, img)
            '''
            kps = np.array(js['keypoints']).reshape(-1,3)[:,:2]
            points = np.vstack( (kps[0], (kps[5] + kps[6])/2, kps[6], kps[8], kps[10], kps[5], kps[7],
            kps[9], kps[12], kps[14], kps[16], kps[11], kps[13], kps[15], kps[2], kps[1], kps[4], kps[3], [0.0, 0.0] ))
            points = [vec for vec in points]
            points = [np.array(vec) for vec in points]
            points_proj = points.copy()
            if [p for p in points if np.any(p == None)]:
                print('None found')
                continue
            BODY_PARTS, POSE_PAIRS = parts(args)
            points = to36M(points, BODY_PARTS)
            points_proj = np.copy(points)
            points = np.reshape(points, [1, -1]).astype('f')
            
            #points_norm = projection_gan.pose.dataset.pose_dataset.pose_dataset_base.Normalization.normalize_2d(points)
            mx, my, scale, points_norm = normalize_2d(points)
            '''
            if first_frame == True:
                pose = create_pose(model, points_norm)
                first_frame = False
                prev_mu_x, prev_mu_y = mu_x, mu_y
            else:
                del_x, del_y = mu_x - prev_mu_x, mu_y - prev_mu_y
                prev_mu_x, prev_mu_y = mu_x, mu_y
                pose = create_pose(model, points_norm)
                pose.T[0::3] += (del_x + prev_hip[0])
                pose.T[1::3] += (del_y + prev_hip[1])
                pose.T[2::3] += (depth_df[depth_df.frame == js['image_id']].z.values[0] * 50)
                prev_hip = pose.T[0:2]
            '''
            pose = create_pose(model, points_norm)
            #depth = depth_df[depth_df.frame == js['image_id']].z.values[0]
            pose_proj = []
            for p in points_proj:
            	pose_proj.append(pixel_to_point(p, 0.5))


            pose.T[0::3] += pose_proj[0][0]
            pose.T[1::3] += pose_proj[0][1]
            pose.T[2::3] += pose_proj[0][2]

            write_to_csv(pose, args, js['image_id'].split('.')[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to image or video. Skip to capture frames from camera')
    parser.add_argument('--proto2d', help='Path to .prototxt', required=True)
    parser.add_argument('--model2d', help='Path to .caffemodel', required=True)
    parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
    parser.add_argument('--inf_engine', action='store_true',
                        help='Enable Intel Inference Engine computational backend. '
                             'Check that plugins folder is in LD_LIBRARY_PATH environment variable')
    parser.add_argument('--lift_model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="COCO")

    parser.add_argument('--activate_func', type=str, default='leaky_relu')
    parser.add_argument('--use_bn', action="store_true")
    parser.add_argument('--mode', type=str, default='video')
    parser.add_argument('--img_folder', type=str, default='./')
    args = parser.parse_args()
    main(args)

    join_csv()
