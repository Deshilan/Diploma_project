import csv
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.widgets as matwid
import os
import copy
import matplotlib.backend_bases as backends

points = [0,0,0,0,0,0,0,0]
def Select(verts):
    global points
    points[0] = int(verts[0][0])
    points[1] = int(verts[0][1])
    points[2] = int(verts[1][0])
    points[3] = int(verts[1][1])
    points[4] = int(verts[2][0])
    points[5] = int(verts[2][1])
    points[6] = int(verts[3][0])
    points[7] = int(verts[3][1])


def writer():
    global points
    file_name = 'test.csv'
    with open(file_name, 'a', encoding='UTF-8') as f:
        CSVWriter = csv.writer(f, lineterminator = '\n')
        CSVWriter.writerow(points)

def Break(event):
    global points
    if event.key == 'e':
        exit()

def Next_rec(event):
    global points
    global PolygonSelector
    if event.key == 'n':
        writer()
        PolygonSelector.disconnect_events()
        app_opening()

def deleter(event):
    if event.key == 'd':
        f = open('test.csv', 'w')
        f.truncate()
        f.close()

fig, ax = plt.subplots()

image = cv2.imread("video.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ax.imshow(image_rgb)

def app_opening():
    global PolygonSelector
    PolygonSelector = matwid.PolygonSelector(ax, Select)
    plt.connect('key_press_event', Next_rec)
    plt.connect('key_press_event', deleter)
    plt.connect('key_press_event', Break)
    plt.show()

app_opening()