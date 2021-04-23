import cv2
import random
import numpy as np
import math

image = cv2.imread('ai_proj.jpg')
# print(image)
# print(len(image), len(image[0]))
# print(image[0][0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# print(image[0][0])

half = int(len(image[0])/2)
left = image[:,:half]
right = image[:,half:]

def euclideanColorDist(a, b):
    print("a", a)
    return math.sqrt( ((a[0]-b[0])**2) + ((a[1] - b[1])**2) + ((a[2] - b[2])**2) )

def euclidDist(a , b):
    dist = np.linalg.norm(a-b)
    return(dist)

def kmeansColors(image):
    representativeColors = []
    # get a random 5 pixels from the image
    # NOTE: find a way to not have duplicates
    x,y = 0,0
    while(len(representativeColors) < 5):
        x = random.randint(0,len(image)-1)
        y = random.randint(0,len(image[0])-1)
        pixel = image[x][y]
        
        representativeColors.append(image[x][y])

    
    print(representativeColors)
    # representativeColors[0][0] = 250
    # print(representativeColors)


    # for each pixel in the image find the closest representative pixel in representativeColors
    # get an average of r,g,b values for each pixel closest to representative pixels in representativeColors
    avgArray = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    numSums = [0,0,0,0,0]
    for i in range(len(image)):
        for j in range(len(image[0])):
            minDist = len(image)*len(image[0])
            minDistColorInd = None
            for x in range(len(representativeColors)):
                newDistance = euclidDist(representativeColors[x], image[i][j])
                if newDistance < minDist:
                    minDist = newDistance
                    minDistColorInd = x
                    
            avgArray[x][0] += representativeColors[x][0]
            avgArray[x][1] += representativeColors[x][1]
            avgArray[x][2] += representativeColors[x][2]
            numSums[x] += 1

    print("avgarr",avgArray)

    # get the averages
    for i in range(5):
        if numSums[i] == 0:
            continue
        avgArray[i][0] /= numSums[i]
        avgArray[i][1] /= numSums[i]
        avgArray[i][2] /= numSums[i]
        # check if r is same as avg r
        if avgArray[i][0] != representativeColors[i][0]:
            representativeColors[i][0] = avgArray[i][0]
        # check if g is same as avg g
        if avgArray[i][1] != representativeColors[i][1]:
            representativeColors[i][1] = avgArray[i][1]
        # check if b is same as avg b
        if avgArray[i][2] != representativeColors[i][2]:
            representativeColors[i][2] = avgArray[i][2]
        
    print(representativeColors)

                    
                
kmeansColors(image)
