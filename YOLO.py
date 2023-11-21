
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

class_name = read_classes("C:\python\pythonProject\YOLO_FILES\ms_coco_classnames.txt")
anchors = read_anchors("C:\python\pythonProject\YOLO_FILES\yolo_anchors.txt")

yolo_model = load_model("")