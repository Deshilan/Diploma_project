import os
import cv2
import random

i = 0
os.chdir('/My_dataset/1')
for filename in os.listdir('/My_dataset/1'):
    sizerand = random.randrange(60, 100, 1)
    size = sizerand/100
    angle = random.randrange(5, 85, 1)
    image = cv2.imread(filename)
    height, width = image.shape[:2]
    rotation = cv2.getRotationMatrix2D((width/2, height/2), angle, size)
    rotated = cv2.warpAffine(image, rotation, (width,height))
    cv2.imwrite("Z_Rot_car" + str(i) + ".jpg" , rotated)
    i = i + 1