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

pliczek = open(r'C:\python\PROJEKT_DYPLOMOWY\Positions\Cars_detection_positions.csv')
plik = pd.read_csv(r'C:\python\PROJEKT_DYPLOMOWY\Positions\Cars_detection_positions.csv', header=None)
i = 0
points = []
image = cv2.imread("HAHA.png")
cv2.imshow('xd', image)
cv2.waitKey(0)
for row in pliczek:
    print(row)
    y1 = int(plik.loc[i, 0])
    x1 = int(plik.loc[i, 1])
    y2 = int(plik.loc[i, 2])
    x2 = int(plik.loc[i, 3])
    image = cv2.rectangle(image, (y1,x1), (y2,x2), color=(0, 0, 255), thickness=5)
    i = i +1
    cv2.imshow('xd', image)
    cv2.waitKey(0)
cv2.imwrite('RESULTS/FNAL.png', image)
