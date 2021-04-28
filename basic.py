import cv2
import random
import numpy as np
import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
from copy import deepcopy
from statistics import mode

# NOTE: height and width are SWITCHED
# NOTE: length of rightGrey is same as length of image??

image = cv2.imread('painting2.jpg')
# print(image)
# print(len(image), len(image[0]))
# print(image[0][0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# print(image[0][0])

half = int(len(image[0])/2)
left = image[:,:half]
right = image[:,half:]

def basicAgent(image):
    # Convert image to grey scale
    greyImg = greyScaleImg(deepcopy(image))
    # plt.imshow(greyImg)
    # plt.show()

    # Use k-means to get the 5 most representative colors in the original image
    colors = kmeansColors(image)
    # Recolor the left side using the 5 most representative colors
    leftColoredImg = recolorLeft(deepcopy(image), colors)

    # Recolor the right side of the image
    rightColoredImg = recolorRight(deepcopy(image), deepcopy(greyImg), leftColoredImg, colors)

    # Combine recolored left and recolored right
    new = []
    for i in range(0, len(leftColoredImg)):
        new.append(list(leftColoredImg[i])+list(rightColoredImg[i]))

    plt.imshow(new)
    plt.show()

def greyScaleImg(image):
    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i][j] = 0.21*image[i][j][0] + 0.72*image[i][j][1] + 0.07*image[i][j][2]
    return image

def recolorLeft(image, representativeColors):
    # Extract the left side of the image to recolor using the 5 most representative colors
    half = int(len(image[0])/2)
    leftSide = image[:,:half]

    for i in range(len(leftSide)):
        for j in range(len(leftSide[0])):
            # Find the cluster that the current pixel belongs to by chosing the centroid closest to it
            minDist = 255**3
            minDistColorInd = None
            for x in range(len(representativeColors)):
                newDistance = euclidDist(representativeColors[x], leftSide[i][j])
                if newDistance < minDist:
                    minDist = newDistance
                    minDistColorInd = x

            # Recolor pixel
            leftSide[i][j][0] = representativeColors[minDistColorInd][0]
            leftSide[i][j][1] = representativeColors[minDistColorInd][1]
            leftSide[i][j][2] = representativeColors[minDistColorInd][2]
    return leftSide

def recolorRight(image, greyImage, leftRecolored, representativeColors):
    # Extract the right side of the grey image
    half = int(len(greyImage[0])/2)
    rightToColor = image[:,half:]
    rightGrey = greyImage[:,half:]
    # Extract the left side of the grey image
    leftGrey = greyImage[:,:half]

    # Get all patches from left side of grey image
    leftPatches = []
    for x in range(1,len(leftGrey)-1):
        for y in range(1,len(leftGrey[0])-1):
            leftPatch = np.array([  [leftGrey[x-1][y-1],leftGrey[x][y-1],leftGrey[x+1][y-1]],
                                    [leftGrey[x-1][y],leftGrey[x][y],leftGrey[x+1][y]],
                                    [leftGrey[x-1][y+1],leftGrey[x][y+1],leftGrey[x+1][y+1]]])
            leftPatches.append(leftPatch)

    print('image:',len(image),"right:",len(rightGrey[0]))
    for i in range(1,len(rightGrey)-1):
        print(i)
        for j in range(1,len(rightGrey[0])-1):
            # get 3x3 gray patch from right side
            rightPatch = np.array([ [rightGrey[i-1][j-1],rightGrey[i][j-1],rightGrey[i+1][j-1]],
                                    [rightGrey[i-1][j],rightGrey[i][j],rightGrey[i+1][j]],
                                    [rightGrey[i-1][j+1],rightGrey[i][j+1],rightGrey[i+1][j+1]]])
            # rightPatch = rightGrey[i-1:i+2,j-1:j+2]
            # for row in rightPatch:
            #     print(row)
            minDistances = [255**3]*6
            minPatches = [None]*6

            # Pick 1000 patches at random, find the best 6 among those
            samples = random.sample(list(leftPatches), 1000)

            for leftPatch in samples:
                newDistance = euclidDist(rightPatch, leftPatch)
                if newDistance < minDistances[0]:
                    minDistances.pop()
                    minDistances.insert(0,newDistance)
                    minPatches.pop()
                    minPatches.insert(0,(x,y))
                elif newDistance < minDistances[1]:
                    minDistances.pop()
                    minDistances.insert(0,newDistance)
                    minPatches.pop()
                    minPatches.insert(0,(x,y))
                elif newDistance < minDistances[2]:
                    minDistances.pop()
                    minDistances.insert(0,newDistance)
                    minPatches.pop()
                    minPatches.insert(0,(x,y))
                elif newDistance < minDistances[3]:
                    minDistances.pop()
                    minDistances.insert(0,newDistance)
                    minPatches.pop()
                    minPatches.insert(0,(x,y))
                elif newDistance < minDistances[4]:
                    minDistances.pop()
                    minDistances.insert(0,newDistance)
                    minPatches.pop()
                    minPatches.insert(0,(x,y))
                elif newDistance < minDistances[5]:
                    minDistances.pop()
                    minDistances.insert(0,newDistance)
                    minPatches.pop()
                    minPatches.insert(0,(x,y))
            # patchColors = []
            # for patch in minPatches:
            #     patchColors.append(leftRecolored[patch[0]][patch[1]])
            # mostFrequent = mode(patchColors)
            # cntr = 0
            # for color in patchColors:
            #     if color == mostFrequent:
            #         cntr+=1
            # if cntr > 3:
            #     rightToColor[i][j] = mostFrequent
            # else:
            #     rightToColor[i][j] = patchColors[0]

            colorCount = [0,0,0,0,0]
            for k in range(len(representativeColors)):
                for patch in minPatches:
                    if (representativeColors[k][0] == leftRecolored[patch[0]][patch[1]][0]) and (representativeColors[k][1] == leftRecolored[patch[0]][patch[1]][1]) and (representativeColors[k][2] == leftRecolored[patch[0]][patch[1]][2]):
                        colorCount[k] += 1
            if max(colorCount) > 3:
                # print("Recoloring at line 145 index : ",(i,j))
                index = colorCount.index(max(colorCount))
                rightToColor[i][j][0] = representativeColors[index][0]
                rightToColor[i][j][1] = representativeColors[index][1]
                rightToColor[i][j][2] = representativeColors[index][2]
            else:
                # print("Recoloring at line 151 index : ",(i,j))
                rightToColor[i][j][0] = leftRecolored[minPatches[0][0]][minPatches[0][1]][0]
                rightToColor[i][j][1] = leftRecolored[minPatches[0][0]][minPatches[0][1]][1]
                rightToColor[i][j][2] = leftRecolored[minPatches[0][0]][minPatches[0][1]][2]
    return rightToColor

def euclideanColorDist(a, b):
    return math.sqrt( ((a[0]-b[0])**2) + ((a[1] - b[1])**2) + ((a[2] - b[2])**2) )

def euclidDist(a , b):
    dist = np.linalg.norm(a-b)
    return(dist)

def kmeansColors(image):
    # Array that hold the 5 random centroids
    representativeColors = []
    
    # NOTE: find a way to not have duplicates
    # Get random 5 pixels from the image
    x,y = 0,0
    while (len(representativeColors) < 5):
        x = random.randint(0,len(image)-1)
        y = random.randint(0,len(image[0])-1)
        pixel = image[x][y]
        
        representativeColors.append(list(image[x][y]))
    # print("Representative Colors at the begining: ",representativeColors)
    # print()

    # for each pixel in the image find the closest representative pixel in representativeColors
    # get an average of r,g,b values for each pixel closest to representative pixels in representativeColors
    avgArray = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    numSums = [0,0,0,0,0]

    for i in range(len(image)):
        for j in range(len(image[0])):
            minDist = 255**3
            minDistColorInd = None
            for x in range(len(representativeColors)):
                newDistance = euclidDist(representativeColors[x], image[i][j])
                if newDistance < minDist:
                    minDist = newDistance
                    minDistColorInd = x

            avgArray[minDistColorInd][0] += image[i][j][0]
            avgArray[minDistColorInd][1] += image[i][j][1]
            avgArray[minDistColorInd][2] += image[i][j][2]
            numSums[minDistColorInd] += 1

    # get the averages
    for i in range(5):
        if numSums[i] == 0:
            continue
        avgArray[i][0] //= numSums[i]
        avgArray[i][1] //= numSums[i]
        avgArray[i][2] //= numSums[i]
        # print("avgarr: ",avgArray)
        # print()
        # check if r is same as avg r
        if abs(avgArray[i][0] - representativeColors[i][0])>5:
            representativeColors[i][0] = avgArray[i][0]
            # print("Replaced at line 75")
        # check if g is same as avg g
        if abs(avgArray[i][1] - representativeColors[i][1])>5:
            representativeColors[i][1] = avgArray[i][1]
            # print("Replaced at line 79")
        # check if b is same as avg b
        if abs(avgArray[i][2] - representativeColors[i][2])>5:
            representativeColors[i][2] = avgArray[i][2]
            # print("Replaced at line 83")
    # print() 
    # print("Representative Colors at the end: ",representativeColors)
    return representativeColors

basicAgent(image)