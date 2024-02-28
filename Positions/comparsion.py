import pandas as pd

recfile =open(r'C:\python\PROJEKT_DYPLOMOWY\Positions\Rectangles.csv')
detectfile = open(r'C:\python\PROJEKT_DYPLOMOWY\Positions\Cars_detection_positions.csv')

PANDA_REC = pd.read_csv('C:\python\PROJEKT_DYPLOMOWY\Positions\Rectangles.csv', header=None)
PANDA_DET = pd.read_csv('C:\python\PROJEKT_DYPLOMOWY\Positions\Cars_detection_positions.csv', header=None)

wynik = 0
suma  = 0
m = 0
n = 0

for row in recfile:
    m = m + 1

for row in detectfile:
    n = n + 1


for x in range(m):
    x1 = PANDA_REC.loc[x, 0]
    x2 = PANDA_REC.loc[x, 2]
    y1 = PANDA_REC.loc[x, 1]
    y2 = PANDA_REC.loc[x, 3]

    for y in range(n):
        xx1 = PANDA_DET.loc[y, 0]
        xx2 = PANDA_DET.loc[y, 2]
        yy1 = PANDA_DET.loc[y, 1]
        yy2 = PANDA_DET.loc[y, 3]

        maxx = max(x1, xx1)
        minx = min(x2, xx2)
        maxy = max(y1, yy1)
        miny = min(y2, yy2)

        if (maxx<minx) and (maxy<miny):
            AO = abs(maxx-minx) * abs(maxy-miny)
        else:
            AO = 0
        print(x2-x1)
        print(y2-y1)
        print(xx2-xx1)
        print(yy2-yy1)
        total = (x2-x1) * (y2-y1) + (xx2-xx1) * (yy2-yy1) - AO
        print(AO, total)
        print(total)
        print(AO/total)
        if (AO/total>0.5):
            wynik = wynik + 1
    suma = suma + 1

print(wynik,suma)