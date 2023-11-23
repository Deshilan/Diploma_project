import os

import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from Diploma_project.yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body, \
    yolo_eval

class_names = read_classes("/home/jan/PycharmProjects/pythonProject/Diploma_project/YOLO_FILES/ms_coco_classnames.txt")
anchors = read_anchors("/home/jan/PycharmProjects/pythonProject/Diploma_project/YOLO_FILES/yolo_anchors.txt")
image_shape = (1000., 1418.)

yolo_model = load_model("/home/jan/PycharmProjects/pythonProject/Diploma_project/YOLO_FILES/yolo.h5")

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    filtering_mask = (box_class_scores > threshold)
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    return scores, boxes, classes

scores, boxes, classes  = yolo_eval(yolo_outputs, image_shape)

def predict(sess, image_file, cv2=None):
    image, image_data = preprocess_image(image_file, model_image_size=(608, 608))
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                    feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    colors = generate_colors(class_names)
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    image.save(os.path.join("out", image_file), quality=90)
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    cv2.imshow(output_image)
    cv2.waitKey(0)

    return out_scores, out_boxes, out_classes

sess = K.get_session()
x,y,z = predict(sess, "/home/jan/PycharmProjects/pythonProject/Diploma_project/baza.jpg")