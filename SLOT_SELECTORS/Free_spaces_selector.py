import csv
import cv2
import matplotlib.pyplot as plt
import matplotlib.widgets as matwid

points = []
x = open('../Positions/Slot_positions.csv', 'w')
x.truncate()
x.close()
def Select(verts):
    global points
    for x in range(len(verts)):
        for y in range(2):
            print(int(verts[x][y]))
            points.append(int(verts[x][y]))

def writer():
    global points
    file_name = '../Positions/Valid_spaces.csv'
    with open(file_name, 'a', encoding='UTF-8') as f:
        CSVWriter = csv.writer(f, lineterminator = '\n')
        CSVWriter.writerow(points)
        points = []

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
        f = open('C:\python\PROJEKT_DYPLOMOWY\Positions\Valid_spaces.csv', 'w')
        f.truncate()
        f.close()

fig, ax = plt.subplots()

image = cv2.imread(r"C:\python\PROJEKT_DYPLOMOWY\Parking_testowy.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
ax.imshow(image_rgb)

def app_opening():
    global PolygonSelector
    PolygonSelector = matwid.PolygonSelector(ax, Select)
    plt.connect('key_press_event', Next_rec)
    plt.connect('key_press_event', deleter)
    plt.connect('key_press_event', Break)
    plt.show()

app_opening()