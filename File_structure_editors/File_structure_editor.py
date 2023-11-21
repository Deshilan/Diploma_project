import os
import shutil


for i in range(9):

    labels = open('C:/python/pythonProject/LABELS/camera' + str(i+1) + '.txt')
    lines = labels.readlines()


    print(lines[0])
    print(lines[0][-2])

    pictures = 'C:/python/pythonProject/PATCHES/SUNNY/2016-01-16/camera' + str(i+1) + '/'
    destination0 = 'C:/python/pythonProject/My_dataset/0/'
    destination1 = 'C:/python/pythonProject/My_dataset/1/'

    for filename in os.listdir(pictures):
        for line in lines:
            if line.find(str(filename)) != -1:
                if line[-2] == '0':
                    shutil.move(pictures + filename, destination0)
                if line[-2] == '1':
                    shutil.move(pictures + filename, destination1)


