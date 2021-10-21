import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json

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

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = '/home/ubuntu/inzamam/videoframes'
total_num_images = len(os.listdir('/home/ubuntu/inzamam/videoframes'))
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'frame{:06d}.jpg'.format(i)) for i in range(0, total_num_images)] 

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
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
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def getcocodict(output_dict, image_path, im_height, im_width):
    x = output_dict['detection_classes']
    detection_boxes_output = output_dict['detection_boxes']
    detection_scores_output =output_dict['detection_scores']
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
    single_image_dict = {}
    single_image_dict["image_id"] = image_path.split('/')[-1]
    single_image_dict["category_id"] = 1
    single_image_dict["bbox"] = bbox
    single_image_dict["score"] = float(max(p_detection_scores))
    return single_image_dict

def getcocovaldict(image_path, im_height, im_width):
    image_dict = {'license': 3, 'file_name': '00000032.jpg', 'coco_url': 
                  'http://images.cocodataset.org/val2017/000000394940.jpg', 'height': 360, 'width': 640, 
                  'date_captured': '203213-11-24 13:47:05', 
                  'flickr_url': 'http://farm9.staticflickr.com/8227/8566023505_e9e9f997bc_z.jpg', 'id': 32}
    image_dict['file_name'] = image_path.split('/')[-1]
    image_dict['height'] = im_height
    image_dict['width'] = im_width
    image_dict['id'] = image_path.split('/')[-1]
    return image_dict

detection_list = []
valimage_dict = {}
for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    detection_list.append(getcocodict(output_dict, image_path, image_np.shape[0], image_np.shape[1]))
    valimage_dict[image_path.split('/')[-1]] = getcocovaldict(image_path, image_np.shape[0], image_np.shape[1])

with open('human_detection.json', 'w') as outfile:
    json.dump(detection_list, outfile)

with open('valimage_dict.json', 'w') as outfile:
    json.dump(valimage_dict, outfile)
