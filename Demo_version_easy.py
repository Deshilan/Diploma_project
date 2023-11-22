import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import cv2
import numpy as np
import pandas as pd
import urllib.request
import time


def save_picture1(image):
    directory = r'C:\python\pythonProject\Temp_image\SLOTS'
    pliczek = open(r'C:\python\pythonProject\test.csv')
    plik = pd.read_csv(r'C:\python\pythonProject\test.csv', header=None)
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
        cv2.imwrite('SLOT_'+str(i)+'.jpg', image1)
        i = i + 1
    return points

def getImage(URL):
    os.chdir("C:\python\pythonProject\Temp_image")
    urllib.request.urlretrieve(URL, "temporary.jpg")

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

model.load_weights(r'C:\python\pythonProject\Upgraded_dataset_model\EXTRA_DATASET_NORMAL_PARAMETERS\my_model.keras')



while(True):
    start_download= time.time()
    URL = "http://96.66.39.30:8090/cgi-bin/faststream.jpg?stream=half&fps=15&rand=COUNTER"
    getImage(URL)
    end_download= time.time()
    download_time = end_download - start_download
    my_part_start = time.time()
    image = cv2.imread(r'C:\python\pythonProject\Temp_image\temporary.jpg')
    points = save_picture1(image)
    directory_es_1 = r'C:\python\pythonProject\Temp_image\SLOTS'
    i = 0
    pola = []
    for filename in os.listdir(directory_es_1):
        path = str(directory_es_1) + '/' + str(filename)
        img = cv2.imread(path)

        img_size = cv2.resize(img, (150, 150))
        new_image = img_size.reshape(-1, 150, 150, 3)

        predict = model.predict([new_image])
        im_class = tf.argmax(predict, axis=-1)
        result = (str(im_class))
        if result[11] == '1':
            pola.append(1)
        if result[11] == '0':
            pola.append(0)
        i = i + 1

    m = 0

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
            cv2.polylines(image, pts=polygon,isClosed=True , color=(255, 0, 0), thickness=3)
        m = m + 1


    cv2.imshow("OBRAZ z kamery", image)
    cv2.waitKey(1)
    number = number + 3
    my_part_end = time.time()
    my_part = my_part_end - my_part_start
    print("CZAS POBIERANIA:" + str(download_time))
    print("CZAS DZIAŁANIA:" + str(my_part))

