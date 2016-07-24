import math

# edge detect functions
def localproduct2D(image, response, xcoor, ycoor):
    sumofproduct = 0

    for yy in range(-1, 2):
        for xx in range(-1, 2):
            sumofproduct = sumofproduct + image[ycoor+yy][xcoor+xx]*response[yy+1][xx+1]
    return sumofproduct

def convolution2D(image, responseX, responseY, imageX, imageY):
    output = [[0 for x in range(imageX)] for y in range(imageY)]
    maxMag = 0
    # eps = 0.0000000001 # 10^-11

    for yy in range(1, imageY-1): # start 1 in from outer edge
        for xx in range(1, imageX-1):
            scalarX = localproduct2D(image, responseX, xx, yy)
            scalarY = localproduct2D(image, responseY, xx, yy)
            magnitude = math.sqrt( scalarX*scalarX + scalarY*scalarY )
            # if we want to know the slope of the edge and have a data structure to support it (e.g. imaginary number)
                # slope = math.atan( scalarY / (scalarX+eps) )
            output[yy][xx] = magnitude
            if magnitude > maxMag:
                maxMag = magnitude
    return output, maxMag

def edgedetect2D(magnitude, thresh, maxMag, imageX, imageY):
    edges = [[0 for x in range(imageX)] for y in range(imageY)]

    for yy in range(imageY):
        for xx in range(imageX):
            magnitudeNorm = magnitude[yy][xx] / maxMag
            if magnitudeNorm >= thresh:
                edges[yy][xx] = 1
    return edges

# initialize image and responses
responseX = [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]

responseY = [[-1,-2,-1],
             [ 0, 0, 0],
             [ 1, 2, 1]]

image = [[0,0.1,0.2,0.3,0.4,0.5,0.6,0.6,0.8,0.9],
         [0,0.1,0.2,0.3,0.4,0.5,0.6,0.6,0.8,0.9],
         [0,0.1,0.2,1.0,1.0,1.0,1.0,0.6,0.8,0.9],
         [0,0.1,0.2,1.0,1.0,1.0,1.0,0.6,0.8,0.9],
         [0,0.1,0.2,1.0,1.0,1.0,1.0,0.6,0.8,0.9],
         [0,0.1,0.2,1.0,1.0,1.0,1.0,0.6,0.8,0.9],
         [0,0.1,0.2,1.0,1.0,1.0,1.0,0.6,0.8,0.9],
         [0,0.1,0.2,1.0,1.0,1.0,1.0,0.6,0.8,0.9],
         [0,0.1,0.2,0.3,0.4,0.5,0.6,0.6,0.8,0.9],
         [0,0.1,0.2,0.3,0.4,0.5,0.6,0.6,0.8,0.9]]

imageY = len(image)
imageX = len(image[0])

# run algorithm
output, maxMag = convolution2D(image, responseX, responseY, imageX, imageY)
edges = edgedetect2D(output, 0.5, maxMag, imageX, imageY)
