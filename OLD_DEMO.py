import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import urllib.request
import time
import keras
import matplotlib
matplotlib.use('Agg')

def save_picture1(image):
    directory = r'/mnt/c/python/PROJEKT_DYPLOMOWY/Temp_image/SLOTS'
    pliczek = open(r'/mnt/c/python/PROJEKT_DYPLOMOWY/Positions/Slot_positions.csv')
    plik = pd.read_csv(r'/mnt/c/python/PROJEKT_DYPLOMOWY/Positions/Slot_positions.csv', header=None)
    os.chdir(directory)
    i = 0
    points = []
    for row in pliczek:
        y1 = int(plik.loc[i, 0])
        x1 = int(plik.loc[i, 1])
        y2 = int(plik.loc[i, 2])
        x2 = int(plik.loc[i, 3])
        y3 = int(plik.loc[i, 4])
        x3 = int(plik.loc[i, 5])
        y4 = int(plik.loc[i, 6])
        x4 = int(plik.loc[i, 7])
        points.append([[y1, x1], [y2, x2], [y3, x3], [y4, x4]])
        minx = min(x1, x2, x3, x4)
        maxx = max(x1, x2, x3, x4)
        miny = min(y1, y2, y3, y4)
        maxy = max(y1, y2, y3, y4)
        mask = np.zeros(image.shape, dtype=np.uint8)
        roi_corners = np.array([(y1, x1), (y2, x2), (y3, x3), (y4, x4)], dtype=np.int32)
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
        cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
        temp_image = cv2.bitwise_and(image, mask)
        image1 = temp_image[minx:maxx, miny:maxy]
        cv2.imwrite('SLOT_'+str(i)+'1.jpg', image1)
        i = i + 1
    return points

def getImage(URL):
    os.chdir("/mnt/c/python/PROJEKT_DYPLOMOWY/Temp_image")
    urllib.request.urlretrieve(URL, "temporary.jpg")


data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(64,
                                  64,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

model_001 = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255, input_shape=(64, 64, 3)),
        layers.Conv2D(32, (6,6), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.0),
        layers.Conv2D(192, (6,6), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(256, (6,6), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(224, activation='relu', input_dim=2),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid'),
    ])


model_001.load_weights('/mnt/c/python/PROJEKT_DYPLOMOWY/Slot occupancy classification models/CNN_STANDARD/Perfect_CNN/my_model.keras')




while(True):

    image = cv2.imread(r'/mnt/c/python/PROJEKT_DYPLOMOWY/Parking_testowy.jpg')
    points = save_picture1(image)
    directory_es_1 = r'/mnt/c/python/PROJEKT_DYPLOMOWY/Temp_image/SLOTS'
    i = 0
    pola = []
    for filename in os.listdir(directory_es_1):
        path = str(directory_es_1) + '/' + str(filename)
        img = cv2.imread(path)

        img_size = cv2.resize(img, (64, 64))
        new_image = img_size.reshape(-1, 64, 64, 3)

        predict = model_001.predict([new_image])
        if predict > 0.5:
            pola.append(1)
        if predict < 0.5:
            pola.append(0)
        i = i + 1

    m = 0
    print(points)
    for polygons in pola:
        y1 = points[m][0][0]
        x1 = points[m][0][1]
        y2 = points[m][1][0]
        x2 = points[m][1][1]
        y3 = points[m][2][0]
        x3 = points[m][2][1]
        y4 = points[m][3][0]
        x4 = points[m][3][1]

        polygon = np.array([[[y1,x1], [y2,x2], [y3,x3], [y4,x4]]], np.int32)
        if pola[m] == 0:
            cv2.polylines(image, pts=polygon, isClosed=True,  color=(0,255,0), thickness=3)
        if pola[m] == 1:
            cv2.polylines(image, pts=polygon,isClosed=True , color=(0, 0, 0), thickness=3)
        m = m + 1
    pola = []
    cv2.imwrite('HAHA.png', image)
