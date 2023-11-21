import PIL
from PIL import Image
import cv2
import os
import random
from PIL import Image, ImageFilter, ImageOps
import tensorflow

directory_em = r'C:\python\pythonProject\Emergency_vehicles_dataset\Emergency'
os.chdir(directory_em)

i = 0

for file in os.listdir(directory_em):
    im = cv2.imread(file)
    im_image = Image.open(file)
    if file[0] == 'Z':
        cropped = im[100:200, 0:200]
        cv2.imwrite("Cropped_" + str(i) + ".jpg", cropped)
        i = i +1 
