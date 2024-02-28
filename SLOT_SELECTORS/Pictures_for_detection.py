import cv2.dnn
import numpy as np
import pandas as pd
import os


def save_picture1():
    directory = r'C:\python\PROJEKT_DYPLOMOWY\Extra_pictures_to_dataset'
    pliczek = open('C:\python\PROJEKT_DYPLOMOWY\Positions\Valid_spaces.csv')
    plik = pd.read_csv('C:\python\PROJEKT_DYPLOMOWY\Positions\Valid_spaces.csv', header=None)
    image = cv2.imread('C:\python\PROJEKT_DYPLOMOWY\Parking_testowy.jpg', -1)
    os.chdir(directory)
    i = 0

    for row in pliczek:

        mask = np.zeros(image.shape, dtype=np.uint8)
        roi_corners = np.array([(y1, x1), (y2, x2), (y3, x3), (y4, x4)], dtype=np.int32)
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
        cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
        temp_image = cv2.bitwise_and(image, mask)
        i = i + 1

    cv2.imwrite("/Temp_image/Detection.jpg", temp_image)

save_picture1()

