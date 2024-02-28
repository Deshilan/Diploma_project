import csv
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.widgets as matwid
import os
import copy
import matplotlib.backend_bases as backends

points = []
def Select(eclick, erealese):
    global points
    xstart = int(eclick.xdata)
    ystart = int(eclick.ydata)
    xend = int(erealese.xdata)
    yend = int(erealese.ydata)
    points = [xstart, ystart, xend, yend]


def writer():
    global points
    file_name = r'C:\python\PROJEKT_DYPLOMOWY\Positions\Rectangles.csv'
    with open(file_name, 'a', encoding='UTF-8') as f:
        CSVWriter = csv.writer(f, lineterminator = '\n')
        CSVWriter.writerow(points)

def Break(event):
    global points
    if event.key == 'e':
        exit()

def Next_rec(event):
    global points
    if event.key == 'n':
        writer()

def deleter(event):
    if event.key == 'd':
        f = open('C:\python\PROJEKT_DYPLOMOWY\Positions\Rectangles.csv', 'w')
        f.truncate()
        f.close()

fig, ax = plt.subplots()

image = cv2.imread("C:\python\PROJEKT_DYPLOMOWY\Vehicle-Detection-main\Detection_test_images\For_test_57.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ax.imshow(image_gray)

RectangleSelector = matwid.RectangleSelector(ax, Select, interactive=True)
plt.connect('key_press_event', Next_rec)
plt.connect('key_press_event', deleter)
plt.show()