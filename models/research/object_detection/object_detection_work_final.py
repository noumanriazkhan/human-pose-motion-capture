import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import cv2

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import multiprocessing
import numpy as np
import glob
import os
from tensorpack.dataflow import BatchData, MultiThreadMapData, PrefetchDataZMQ, dataset, ImageFromFile, TestDataSpeed
from tensorpack import dataflow

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = '/home/ubuntu/inzamam/videoframes'
num_frames = len(os.listdir(PATH_TO_TEST_IMAGES_DIR)) - 1
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'frame{:06d}.jpg'.format(i)) for i in range(0, len(os.listdir(PATH_TO_TEST_IMAGES_DIR)) - 1) ]

ops = detection_graph.get_operations()
all_tensor_names = {output.name for op in ops for output in op.outputs}
tensor_dict = {}
for key in [
    'num_detections', 'detection_boxes', 'detection_scores',
    'detection_classes', 'detection_masks'
]:
  tensor_name = key + ':0'
  if tensor_name in all_tensor_names:
    tensor_dict[key] = detection_graph.get_tensor_by_name(
        tensor_name)
if 'detection_masks' in tensor_dict:
  # The following processing is only for single image
  detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
  detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
  # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
  real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
  detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
  detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
  detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
      detection_masks, detection_boxes, image.shape[1], image.shape[2])
  detection_masks_reframed = tf.cast(
      tf.greater(detection_masks_reframed, 0.5), tf.uint8)
  # Follow the convention by adding back the batch dimension
  tensor_dict['detection_masks'] = tf.expand_dims(
    detection_masks_reframed, 0)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
sess = tf.Session(graph = detection_graph)


def getdataFlowObject(list_of_images, batch_size, parallel=None):
    df = ImageFromFile(files = list_of_images, channel=3, resize=None, shuffle=False)
    df = BatchData(df, batch_size, remainder=True)
    df = PrefetchDataZMQ(df, 2)
    return df

df = getdataFlowObject(TEST_IMAGE_PATHS, batch_size=128)
df.reset_state()

for dp in df:
    output_dict = sess.run(tensor_dict, feed_dict = {image_tensor: dp[0]})
