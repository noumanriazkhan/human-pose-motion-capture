import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import cv2
import glob
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

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
#num_frames = len(os.listdir(PATH_TO_TEST_IMAGES_DIR)) - 1
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'frame{:06d}.jpg'.format(i)) for i in range(0, len(os.listdir(PATH_TO_TEST_IMAGES_DIR)) - 1) ]
TEST_IMAGE_PATHS = glob.glob( os.path.join (PATH_TO_TEST_IMAGES_DIR, '*.jpg'))
num_frames = len(TEST_IMAGE_PATHS)
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

# inference related code starts from here
def parse_function(filename):
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.uint8)
    return filename, image

def getFrames(video_name, folder_name="", verbose = False):
    frames = np.zeros([424,720,1080,3], dtype=np.uint8)
    vidcap = cv2.VideoCapture(video_name)
    success,image = vidcap.read()
    if success == False:
        print('No video file found or no supported format!')
    count = 0
    
    while success:
#         cv2.imwrite(os.path.join(folder_name, "frame%06d.jpg" % count), image)     # save frame as JPEG file
        frames[count,:,:,:] = image
        success,image = vidcap.read()
        if verbose == True:
            print('Read a new frame: ', success)
        count += 1
    return frames


batch_size = 128

dataset = tf.data.Dataset.from_tensor_slices((TEST_IMAGE_PATHS))
dataset = dataset.map(parse_function, num_parallel_calls=8)

dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)

iterator = dataset.make_one_shot_iterator()
file_name, next_element = iterator.get_next()
graph_2 = tf.Graph()
sess2 = tf.Session()


i = 0
output_dict = {}
while True:
    try:
        elem = sess2.run(next_element)
        print(np.shape(elem))
        output_dict[i] = sess.run(tensor_dict, feed_dict = {image_tensor: elem})
        i = i + 1
    except tf.errors.OutOfRangeError:
        print("End of training dataset.")
        break
        
def getcocodict(output_dict, image_path, im_height, im_width, batch_size):
    batch_image_list = []
    for i in range(batch_size):
        x = output_dict['detection_classes'][i]
        detection_boxes_output = output_dict['detection_boxes'][i]
        detection_scores_output =output_dict['detection_scores'][i]
        person_indices = np.where(x == 1)
        p_detection_boxes = detection_boxes_output[person_indices]
        p_detection_scores = detection_scores_output[person_indices]
        ymin = p_detection_boxes[np.where(max(p_detection_scores))][0][0]
        xmin = p_detection_boxes[np.where(max(p_detection_scores))][0][1]
        ymax = p_detection_boxes[np.where(max(p_detection_scores))][0][2]
        xmax = p_detection_boxes[np.where(max(p_detection_scores))][0][3]

        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)

        bbox = [left, top, right - left, bottom - top]
        image_dict = {}
        image_dict["image_id"] = image_path[i].split('/')[-1]
        image_dict["category_id"] = 1
        image_dict["bbox"] = bbox
        image_dict["score"] = float(max(p_detection_scores))
        batch_image_list.append(image_dict)
    return batch_image_list

def getcocovaldict(image_path, im_height, im_width, batch_size):
    val_list = []
    batch_dict = {}
    for i in range(batch_size):
        image_dict = {'license': 3, 'file_name': '00000032.jpg', 'coco_url': 
                      'http://images.cocodataset.org/val2017/000000394940.jpg', 'height': 360, 'width': 640, 
                      'date_captured': '203213-11-24 13:47:05', 
                      'flickr_url': 'http://farm9.staticflickr.com/8227/8566023505_e9e9f997bc_z.jpg', 'id': 32}
        image_dict['file_name'] = image_path[i].split('/')[-1]
        image_dict['height'] = im_height
        image_dict['width'] = im_width
        image_dict['id'] = image_path[i].split('/')[-1]
        batch_dict[image_path[i].split('/')[-1]] = image_dict
    return batch_dict

image = Image.open(TEST_IMAGE_PATHS[0])
(im_width, im_height) = image.size

detection_list = []
valimage_dict = {}
for i in range(len(output_dict)):
    start = i * batch_size
    end = i * batch_size +  batch_size
    if end >= len(TEST_IMAGE_PATHS):
        end = len(TEST_IMAGE_PATHS)
        batch_size = end - start
    print(start, end, batch_size)
    detection_list.extend(getcocodict(output_dict = output_dict[i],image_path = TEST_IMAGE_PATHS[start:end], 
                                      im_height= im_height,im_width= im_width, batch_size= batch_size))
    valimage_dict.update(getcocovaldict(image_path = TEST_IMAGE_PATHS[start:end], im_height=im_height, 
                                        im_width=im_width, batch_size=batch_size))

with open('human_detection.json', 'w') as outfile:
    json.dump(detection_list, outfile)

with open('valimage_dict.json', 'w') as outfile:
    json.dump(valimage_dict, outfile)
