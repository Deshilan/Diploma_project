import os
import PIL.Image as Image
import cv2

dir = r'C:\python\PROJEKT_DYPLOMOWY\NEW_EM_DATASET\TEST\331\0'
os.chdir(dir)

for file in os.listdir(dir):
    image = Image.open(file)
    new = image.resize((331,331))
    new.save(file)

dir = r'C:\python\PROJEKT_DYPLOMOWY\NEW_EM_DATASET\TEST\331\1'
os.chdir(dir)

for file in os.listdir(dir):
    image = Image.open(file)
    new = image.resize((331,331))
    new.save(file)