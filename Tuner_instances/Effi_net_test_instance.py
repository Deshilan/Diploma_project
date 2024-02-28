import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import cv2
import numpy as np

model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(150, 150, 3),
    pooling=None,
    classes=2,
    classifier_activation='relu'
)

old_model =  keras.models.load_model(r'/mnt/c/python/PROJEKT_DYPLOMOWY/my_model.keras')

weights = old_model.get_weights()
model.set_weights(weights)


directory_es_0 = r'/mnt/c/python/PROJEKT_DYPLOMOWY/Part_images/FULL_TEST_DATASET/0'
directory_es_1 = r'/mnt/c/python/PROJEKT_DYPLOMOWY/Part_images/FULL_TEST_DATASET/1'


sum = 0
sum_good_decision = 0
##################################################################

i = 0
good_decision = 0

for filename in os.listdir(directory_es_1):
    path = str(directory_es_1) +'/'+ str(filename)
    img = cv2.imread(path)

    img_size = cv2.resize(img, (150, 150))
    new_image = img_size.reshape(-1, 150, 150, 3)


    predict = model.predict([new_image])
    im_class = tf.argmax(predict, axis = -1)
    result = (str(im_class))
    if result[11] == '1':
        good_decision = good_decision + 1
    i = i + 1

print(str(good_decision) + '/' + str(i))

sum = sum + i
sum_good_decision = sum_good_decision + good_decision

##################################################################

i = 0
good_decision = 0

for filename in os.listdir(directory_es_0):
    path = str(directory_es_0) +'/'+ str(filename)
    img = cv2.imread(path)

    img_size = cv2.resize(img, (150, 150))
    new_image = img_size.reshape(-1, 150, 150, 3)


    predict = model.predict([new_image])
    im_class = tf.argmax(predict, axis = -1)
    result = (str(im_class))
    if str(result[11]) == '0':
        good_decision = good_decision + 1
    i = i + 1

print(str(good_decision) + '/' + str(i))
sum = sum + i
sum_good_decision = sum_good_decision + good_decision
