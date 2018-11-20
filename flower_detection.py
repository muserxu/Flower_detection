
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[1]:

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

from utils import label_map_util

from utils import visualization_utils as vis_util

def crop(image_path):

  # What model to use.
  MODEL_NAME = 'inference_graph'

  # Path to frozen detection graph. This is the actual model that is used for the object detection.
  PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

  # List of the strings that is used to add correct label for each box.
  CWD_PATH = os.getcwd()
  PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

  # load model
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  # loading label map
  category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


  # In[11]:
  string_idx = 0

  # read the image
  image_np = cv2.imread(image_path)

  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)

  # finding the smallest box
  min = 1
  index = 0
  for i in range(output_dict['num_detections']):
    if min > (output_dict['detection_boxes'][i][2]-output_dict['detection_boxes'][i][0]) * (output_dict['detection_boxes'][i][3]-output_dict['detection_boxes'][i][1]):
      min = (output_dict['detection_boxes'][i][2]-output_dict['detection_boxes'][i][0]) * (output_dict['detection_boxes'][i][3]-output_dict['detection_boxes'][i][1])
      index = i

  detection_box = np.array([output_dict['detection_boxes'][index]])
  detection_classes = np.array([output_dict['detection_classes'][index]])
  detection_scores = np.array([output_dict['detection_scores'][index]])

  cols,rows,ch = image_np.shape
  y1 = int(round(detection_box[0][0] *cols))
  x1 = int(round(detection_box[0][1] *rows))
  y2 = int(round(detection_box[0][2] *cols))
  x2 = int(round(detection_box[0][3] *rows))
  # print (x1,x2,y1,y2)
  crop_img = image_np[y1:y2, x1:x2]
  # cv2.imshow('ss', cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
  # cv2.waitKey(0)
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      detection_box,
      detection_classes,
      detection_scores,
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8, max_boxes_to_draw=2,
      min_score_thresh=0.85) 

  # cv2.waitKey(0)
  string1 = './detected_image/detected' +str(string_idx)+'.jpg'
  string2 = './detected_image/cropped' +str(string_idx)+'.jpg'

  # write cropped image to file
  cv2.imwrite(string1, image_np)
  cv2.imwrite(string2, crop_img)
  string_idx+=1

  crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
  return crop_img

# helper function
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
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

if __name__ == '__main__':
  crop('test_images/3.png')


