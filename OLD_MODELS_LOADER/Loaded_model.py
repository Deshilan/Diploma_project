import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import cv2
import numpy as np

model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(150, 150, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2)
    ])



model.load_weights(r'C:\python\PROJEKT_DYPLOMOWY\my_model.keras')


directory_es_0 = r'C:\python\PROJEKT_DYPLOMOWY\Part_images\Easy_dataset\0'
directory_es_1 = r'C:\python\PROJEKT_DYPLOMOWY\Part_images\Easy_dataset\1'
directory_diff_0 = r'C:\python\PROJEKT_DYPLOMOWY\Part_images\Difficult_dataset\0'
directory_diff_1 = r'C:\python\PROJEKT_DYPLOMOWY\Part_images\Difficult_dataset\1'

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

##################################################################

i = 0
good_decision = 0
for filename in os.listdir(directory_diff_1):
    path = str(directory_diff_1) +'/'+ str(filename)
    img = cv2.imread(path)

    img_size = cv2.resize(img, (150, 150))
    new_image = img_size.reshape(-1, 150, 150, 3)


    predict = model.predict([new_image])
    im_class = tf.argmax(predict, axis = -1)
    result = (str(im_class))
    if str(result[11]) == "1":
        good_decision = good_decision + 1
    i = i + 1

print(str(good_decision) + '/' + str(i))

sum = sum + i
sum_good_decision = sum_good_decision + good_decision

##################################################################

i = 0
good_decision = 0
for filename in os.listdir(directory_diff_0):
    path = str(directory_diff_0) +'/'+ str(filename)
    img = cv2.imread(path)

    img_size = cv2.resize(img, (150, 150))
    new_image = img_size.reshape(-1, 150, 150, 3)


    predict = model.predict([new_image])
    im_class = tf.argmax(predict, axis = -1)
    result = (str(im_class))
    if str(result[11]) =="0":
        good_decision = good_decision + 1
    i = i + 1

print(str(good_decision) + '/' + str(i))
sum = sum + i
sum_good_decision = sum_good_decision + good_decision

##################################################################


print(str(sum_good_decision) + "/" + str(sum))