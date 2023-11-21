import PIL
from PIL import Image
import cv2
import os
import random


directory = r'C:\python\pythonProject\Extra_pictures_to_dataset\0'
os.chdir(directory)

i = 0
for file in os.listdir(directory):

    image = Image.open(file)
    sizerand = random.randrange(60, 100, 1)
    size = sizerand/100
    angle = random.randrange(5, 20, 1)
    image = cv2.imread(file)
    height, width = image.shape[:2]
    rotation = cv2.getRotationMatrix2D((width/2, height/2), angle, size)
    rotated = cv2.warpAffine(image, rotation, (width,height))
    cv2.imwrite("EXTRA_DATA5-20_Rot_car" + str(i) + ".jpg" , rotated)

    sizerand = random.randrange(60, 100, 1)
    size = sizerand/100
    angle = random.randrange(20, 35, 1)
    image = cv2.imread(file)
    height, width = image.shape[:2]
    rotation = cv2.getRotationMatrix2D((width/2, height/2), angle, size)
    rotated = cv2.warpAffine(image, rotation, (width,height))
    cv2.imwrite("EXTRA_DATA20-35_Rot_car" + str(i) + ".jpg" , rotated)

    sizerand = random.randrange(60, 100, 1)
    size = sizerand/100
    angle = random.randrange(35, 50, 1)
    image = cv2.imread(file)
    height, width = image.shape[:2]
    rotation = cv2.getRotationMatrix2D((width/2, height/2), angle, size)
    rotated = cv2.warpAffine(image, rotation, (width,height))
    cv2.imwrite("EXTRA_DATA35-50_Rot_car" + str(i) + ".jpg" , rotated)

    sizerand = random.randrange(60, 100, 1)
    size = sizerand / 100
    angle = random.randrange(50, 65, 1)
    image = cv2.imread(file)
    height, width = image.shape[:2]
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, size)
    rotated = cv2.warpAffine(image, rotation, (width, height))
    cv2.imwrite("EXTRA_DATA50-65_Rot_car" + str(i) + ".jpg", rotated)

    sizerand = random.randrange(60, 100, 1)
    size = sizerand / 100
    angle = random.randrange(65, 85, 1)
    image = cv2.imread(file)
    height, width = image.shape[:2]
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, size)
    rotated = cv2.warpAffine(image, rotation, (width, height))
    cv2.imwrite("EXTRA_DATA65-85_Rot_car" + str(i) + ".jpg", rotated)

    i = i + 1