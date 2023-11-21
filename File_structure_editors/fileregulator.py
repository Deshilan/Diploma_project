import cv2.dnn
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt

def save_picture():
    image = Image.open('parking_testowy.jpg')
    pliczek = open('../test.csv')
    plik = pd.read_csv('../test.csv', header=None)
    i = 0
    for row in pliczek:
        image = image[int(plik.loc[i, 0]): int(plik.loc[i, 2]), int(plik.loc[i, 1]):  int(plik.loc[i, 2])]
        i = i +1
        image.save('C:\python\pythonProject\Part_images')


def save_picture1():
    directory = 'C:\python\pythonProject\Part_images'
    pliczek = open('../test.csv')
    plik = pd.read_csv('../test.csv', header=None)
    image = cv2.imread('parking_testowy.jpg')
    os.chdir(directory)
    i = 0
    for row in pliczek:
        ystart = int(plik.loc[i, 0])
        xstart = int(plik.loc[i, 1])
        yend = int(plik.loc[i, 2])
        xend = int(plik.loc[i, 3])
        image1 = image[xstart: xend, ystart:  yend]
        cv2.imshow('new win', image1)
        cv2.waitKey(0)
        cv2.imwrite('Slot_'+str(i)+'.jpg', image1)
        i = i + 1

save_picture1()