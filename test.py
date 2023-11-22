import numpy
x1 = 0
y1 = 1
x2 = 2
y2 = 2
x3 = 3
y3 = 4
x4 = 5
y4 = 6
points = numpy.array([])
points = numpy.append(points, [x1, y1], axis=1)
points = numpy.append(points, [x2, y2], axis=1)
points = numpy.append(points, [x3, y3], axis=1)
points = numpy.append(points, [x4, y4], axis=1)
print(points)