import os
import cv2
import pandas as pd
import csv

plik = open('C:\_annotations.csv')


m = 0
for row in plik:
    x = row.split(',')
    x[7].rstrip()
    print(x[7])
    print(x)
    x1 = int(x[4])
    x2 = int(x[6])
    y1 = int(x[5])
    y2 = int(x[7])
    print(x1,x2,y1,y2)
    for picture in os.listdir("C:/valid/"):
        os.chdir('C:/valid')
        if picture == x[0]:
            if x[3] == 'straz pozarna' or x[3] == 'policja' or x[3] == 'ambulans' or x[3] == 'karetka':
                image = cv2.imread(picture)
                new_img = image[y1:y2, x1:x2]
                os.chdir(r'C:\python\PROJEKT_DYPLOMOWY\NEW_EM_DATASET\Emergency')
                cv2.imwrite("PICV"+str(m)+".jpg", new_img)
                m = m + 1
            else:
                image = cv2.imread(picture)
                new_img = image[y1:y2, x1:x2]
                os.chdir(r'C:\python\PROJEKT_DYPLOMOWY\NEW_EM_DATASET\non_em')
                cv2.imwrite("PICV"+str(m)+".jpg", new_img)
                m = m + 1