"""
Usage:
  python generate_tfrecord.py --labels=annotation.txt  --output_path=train.record --image_dir=./train/

  python generate_tfrecord.py --labels=annotation.txt  --output_path=test.record --image_dir=./test/
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import io
import tensorflow as tf

import src.dataset_utils as dataset_util

flags = tf.app.flags
flags.DEFINE_string('labels', 'annotation.txt', 'Path to the input txt file')
flags.DEFINE_string('output_path', './train.record', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', './train/', 'Path to images')
FLAGS = flags.FLAGS

with open(FLAGS.labels, 'r') as fp:
    data = fp.readlines()
images = {}
for i in data:
    images[i.split(' ')[0]] = i.split(' ')[1:]

def create_tf_example(name):
  filename =  FLAGS.image_dir + name
  print(filename)
  with tf.gfile.GFile((filename), 'rb') as fid:
    encoded_jpg = fid.read()

  img=cv2.imread(filename)
  filename = filename.encode('utf8')
  encoded_image_data = io.BytesIO(encoded_jpg)
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List ofrmalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  height,width,channel = img.shape
  ann = images[name]
  for j in range(int(ann[0])):
  	xmins.append((float(ann[(j*5+1)]))/width)
  	ymins.append((float(ann[(j*5+2)]))/height)
  	xmaxs.append((float(int(ann[(j*5+1)])+int(ann[(j*5+3)])))/width)
  	ymaxs.append((float(int(ann[(j*5+2)])+int(ann[(j*5+4)])))/height)
  	classes_text.append("product".encode('utf8'))
  	classes.append(1)
  print(filename)
  print(len(xmins), len(ymins), len(xmaxs), len(ymaxs), len(classes_text), len(classes))
  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_feature(classes_text),
      'image/object/class/label': dataset_util.int64_feature(classes),
  }))
  return tf_example

def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    filenames = [i for i in os.listdir(FLAGS.image_dir) if i.endswith('.JPG')]
    for filename in filenames:
        tf_example = create_tf_example(filename)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
