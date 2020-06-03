#python infilect_test.py --gt=annotation.txt --test_dir=./tfrecords_prepare/test/ --pb=./single_27855/frozen_inference_graph.pb --out_dir=./test_out --save_im=0

import os
import cv2
import json
import argparse
import numpy as np
import tensorflow as tf
from src.metrics import get_avg_precision_at_iou


ap = argparse.ArgumentParser()
ap.add_argument("--gt", required=True, help="grount truth values of test data")
ap.add_argument("--test_dir", required=True, help="test data set images directory")
ap.add_argument("--pb", required=True, help="trained pd model path")
ap.add_argument("--out_dir", required=True, help="output directory")
ap.add_argument("--save_im", required=True, help="saving of drawn images")
args = vars(ap.parse_args())

with open(args["gt"], 'r') as fp:
    data = fp.readlines()
images = {}
for i in data:
    images[i.split(' ')[0]] = i.split(' ')[1:]

labels = {}
predict = {}
image2products = {}
def detect_objects(image_np, name, sess):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    h , w, c = image_np.shape
    #ipdb.set_trace()
    pbox = []
    pscore = []
    count = 0
    for i in range(scores.shape[1]):    
      if(scores[0][i] > 0.70):
        count = count + 1
        ymin = int(boxes[0][i][0]*h)
        xmin = int(boxes[0][i][1]*w)
        ymax = int(boxes[0][i][2]*h)
        xmax = int(boxes[0][i][3]*w)
        pbox.append([xmin, ymin, xmax, ymax])
        pscore.append(float(scores[0][i]))
        if(int(args["save_im"])):
          cv2.rectangle(image_np,(xmin, ymin),(xmax, ymax),(0,255,0),3)
          cv2.putText(image_np, str(scores[0][i]), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    predict[name]={"boxes":pbox, "scores":pscore}
    image2products[name] = count
    if(int(args["save_im"])):
      cv2.imwrite(os.path.join(args["out_dir"],"out_images",name), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    
def main():
    global detection_graph
    if not os.path.exists(args["out_dir"]):
        os.mkdir(args["out_dir"])
    if(int(args["save_im"])):
      if not os.path.exists(os.path.join(args["out_dir"],"out_images")):
        os.mkdir(os.path.join(args["out_dir"],"out_images"))
    detection_graph = tf.Graph()
    sess = tf.Session(graph=detection_graph)
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args["pb"], 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    filenames = [j for j in os.listdir(args["test_dir"]) if j.endswith('.JPG')]
    for i in filenames:
      img = cv2.imread(os.path.join(args["test_dir"],i))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      output = detect_objects(img ,i, sess)
      gbox = []
      ann = images[i]
      for j in range(int(ann[0])):
        xmin=(float(ann[(j*5+1)]))
        ymin=(float(ann[(j*5+2)]))
        xmax=(float(int(ann[(j*5+1)])+int(ann[(j*5+3)])))
        ymax=(float(int(ann[(j*5+2)])+int(ann[(j*5+4)])))
        gbox.append([xmin, ymin, xmax, ymax])
        labels[i] = gbox
    sess.close()
    #with open('gt.json', 'w') as fp:
    #  json.dump(labels, fp)
    #with open('predict.json', 'w') as fp:
    #  json.dump(predict, fp)
    data = get_avg_precision_at_iou(labels, predict, iou_thr=0.7)
    metrics = {'mAP':data['avg_prec'], 'precision':np.mean(data['precisions']), 'recall': np.mean(data['recalls'])}
    with open(os.path.join(args["out_dir"],'image2products.json'), 'w') as fp:
      json.dump(image2products, fp)
    with open(os.path.join(args["out_dir"],'metrics.json'), 'w') as fp:
      json.dump(metrics, fp)

if __name__ == '__main__':
   main()
