import cv2.dnn
import numpy as np
import pandas as pd
import os


def save_picture1():
    directory = r'C:\python\pythonProject\Extra_pictures_to_dataset'
    pliczek = open('test.csv')
    plik = pd.read_csv('test.csv', header=None)
    image = cv2.imread('Test_images/For_test_61.jpg', -1)
    os.chdir(directory)
    i = 0
    for row in pliczek:
        y1 = int(plik.loc[i, 0])
        x1 = int(plik.loc[i, 1])
        y2 = int(plik.loc[i, 2])
        x2 = int(plik.loc[i, 3])
        y3 = int(plik.loc[i, 4])
        x3 = int(plik.loc[i, 5])
        y4 = int(plik.loc[i, 6])
        x4 = int(plik.loc[i, 7])
        minx = min(x1, x2, x3, x4)
        maxx = max(x1, x2, x3, x4)
        miny = min(y1, y2, y3, y4)
        maxy = max(y1, y2, y3, y4)
        mask = np.zeros(image.shape, dtype=np.uint8)
        print(mask)
        roi_corners = np.array([(y1, x1), (y2, x2), (y3, x3), (y4, x4)], dtype=np.int32)
        print(roi_corners)
        channel_count = image.shape[2]
        print(channel_count)
        ignore_mask_color = (255,) * channel_count
        print(ignore_mask_color)
        cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
        temp_image = cv2.bitwise_and(image, mask)
        image1 = temp_image[minx:maxx, miny:maxy]
        cv2.imwrite('Slot_'+str(i)+'_New_61.jpg', image1)
        i = i + 1

save_picture1()