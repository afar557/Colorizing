import cv2
import random
import numpy as np
import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
from copy import deepcopy
from statistics import mode
import sys

image = cv2.imread('ai_proj_2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def getRandWeights():
    out1W = np.random.rand(6,10)
    out2W = np.random.rand(3,6)
    return out1W, out2W

def sigmoid(dotProdArr):
    computedOut = 1/(1 + np.exp(-dotProdArr))
    return computedOut

def derivSigmoid(dotProdArr):
    computedOut = np.exp(-dotProdArr)/np.square(1 + np.exp(-dotProdArr))
    return computedOut

def getOutputFromModel(out1W, out2W, patch):
    # out1W size is 5x10
    # print("out1W is size:", out1W.shape, out1W)
    # print()

    # out2W size is 3x6
    # print("out2W is size:", out2W.shape, out2W)
    # print()

    # Patch size is 10x3
    # print("patch is size:", patch.shape, patch)
    # print()
    
    # print("Transpose patch is size:", np.transpose(patch).shape, np.transpose(patch))
    # print()

    # out1 size is 6x1
    out1 = sigmoid(np.dot(out1W,np.transpose(patch)))
    # out1 = np.insert(out1, 0, 1, axis=0)
    # print("out1 is size:", out1.shape, out1)
    # print()

    # out2 size is 3x1
    out2 = sigmoid(np.dot(out2W,out1))
    # print("out2 is size:", out2.shape, out2)
    # print()
    return out1, out2

def lossFunc(out2, image, middlePixInd):
    x,y = middlePixInd
    loss = (out2[0] - image[x][y][0])**2 + (out2[1] - image[x][y][1])**2 + (out2[2] - image[x][y][2])**2
    return loss

def updateWeights(out1W, out2W, out1, out2, image, middlePixInd, patch):
    x,y = middlePixInd
    middlePix = image[x][y]

    #Calculate weight2
    # derivLoss size is 3x3
    derivLoss = 2*np.diagflat(np.transpose(out2) - np.array([middlePix]))
    # print("derivLoss size is: ", derivLoss.shape)

    # sigPrime size is 3x1
    sigPrime = derivSigmoid(np.dot(out2W, out1))
    # print("sigPrime size is: ", sigPrime.shape)

    # dotProd size is 3x1
    dotProd = np.dot(derivLoss,sigPrime)
    # print("dotProd size is: ", dotProd.shape)
    
    # newWeight2 size is 3x6
    newWeight2 = np.dot(dotProd,np.transpose(out1))
    # print("newWeight1 size is: ", newWeight2.shape)
    
    # Calculate weight1
    # Calculate the deriv term by doing a sum
    derivLoss2 = np.array([0.0]*6)
    for i in range(3):
        derivLoss3 = 2*(out2[i] - middlePix[i])
        # print("derivLoss3 :", derivLoss3.shape, derivLoss3)

        sigPrime2 = derivSigmoid(np.dot(out2W[i],out1))
        # print("sigPrime2 :", sigPrime2.shape, sigPrime2)

        prod = derivLoss3 * sigPrime2
        # print("prod :", prod.shape, prod)

        derivLoss2 += (prod * out2W[i])
        # print("out2W[i] :", out2W[i].shape, out2W[i])

    # 6x6
    derivLoss2 = np.diagflat(derivLoss2)
    # print("derivLoss2 :", derivLoss2.shape, derivLoss2)

    # 5x1
    sigPrime3 = derivSigmoid(np.dot(out1W, np.transpose(patch)))
    # print("sigPrime3 :", sigPrime3.shape, sigPrime3)

    # 5x10
    dotProd2 = np.dot(sigPrime3,patch)
    # print("dotProd2 :", dotProd2.shape, dotProd2)

    # 
    newWeight1 = np.dot(derivLoss2,dotProd2)
    # print("newWeight1 :", newWeight1.shape, newWeight1)

    return newWeight1, newWeight2

def train(leftGrey, image):
    leftPatches = []
    for x in range(1,len(leftGrey)-1):
        for y in range(1,len(leftGrey[0])-1):
            leftPatch = np.array([[1,leftGrey[x-1][y-1],leftGrey[x][y-1],leftGrey[x+1][y-1],leftGrey[x-1][y],leftGrey[x][y],leftGrey[x+1][y],leftGrey[x-1][y+1],leftGrey[x][y+1],leftGrey[x+1][y+1]]])
            leftPatch/=255
            leftPatches.append([leftPatch,(x,y)])

    # get initial random weights
    out1W, out2W = getRandWeights()
    # print("out1W is :", out1W)
    # print()
    # print("out2W is :", out2W)
    # print()

    # pick  random patch and plug in patch into the model
    cntr = 0
    prevLoss = sys.maxsize
    alpha = 1
    while(True):
        print(cntr)
        x = random.randint(0,len(leftPatches)-1)
        out1, out2 = getOutputFromModel(out1W, out2W, leftPatches[x][0])
        # print("out1 is :", out1)
        # print()
        # print("out2 is :", out2)
        # print()

        loss = lossFunc(out2, image, leftPatches[x][1])
        # print("loss is :", loss)

        newWeight1, newWeight2 = updateWeights(out1W, out2W, out1, out2, image, leftPatches[x][1], leftPatches[x][0])
        # print("newWeight1 :", newWeight1)
        # print("newWeight2 :", newWeight2)

        out1W -= (alpha*newWeight1)
        out2W -= (alpha*newWeight2)
        cntr += 1
        if cntr == 1000:
            change = abs(prevLoss - loss) / prevLoss
            if change > .1:
                prevLoss = loss
                cntr = 0
                continue
            else:
                break
    return out1W, out2W
        
def recolorRight(image, rightGrey, out1W, out2W):
    half = int(len(image[0])/2)
    rightToColor = image[:,half:]
    rightToColor = np.array(rightToColor)
    print("out1w: ", out1W)
    print("out2w: ", out2W)
    for i in range(1,len(rightGrey)-1):
        for j in range(1,len(rightGrey[0])-1):
            # get 3x3 gray patch from right side
            rightPatch = np.array([[1,rightGrey[i-1][j-1],rightGrey[i][j-1],rightGrey[i+1][j-1],rightGrey[i-1][j],rightGrey[i][j],rightGrey[i+1][j],rightGrey[i-1][j+1],rightGrey[i][j+1],rightGrey[i+1][j+1]]])
            rightPatch/=255
            # rightPatch = np.array([ [rightGrey[i-1][j-1],rightGrey[i][j-1],rightGrey[i+1][j-1]],
            #                         [rightGrey[i-1][j],rightGrey[i][j],rightGrey[i+1][j]],
            #                         [rightGrey[i-1][j+1],rightGrey[i][j+1],rightGrey[i+1][j+1]]])
            print("right patch is: ", rightPatch)
            out2 = getOutputFromModel(out1W, out2W, rightPatch)[1]
            print("out2 patch is: ", out2)
            print()
            newR, newG, newB = out2

            rightToColor[i][j][0] = newR*255
            print("multip", newR*255)
            print("righttocolor", rightToColor[i][j][0])
            rightToColor[i][j][1] = newG*255
            rightToColor[i][j][2] = newB*255
    return rightToColor

# Function that turns an image into grey scale
def greyScaleImg(image):
    image = image.tolist()
    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i][j] = 0.21*image[i][j][0] + 0.72*image[i][j][1] + 0.07*image[i][j][2]
    return np.array(image)

def improvedAgent(image):
    # Convert image to grey scale
    greyImg = greyScaleImg(deepcopy(image))
    half = int(len(image[0])/2)
    leftColoredImg = image[:,:half]
    # rightColorImg = img[:,half:]
    leftGrey = greyImg[:,:half]
    rightGrey = greyImg[:,half:]
    out1W, out2W = train(leftGrey, image)
    rightColoredImg = recolorRight(deepcopy(image), rightGrey, out1W, out2W)
    # Combine the recolored right and original left image
    new = []
    for i in range(0, len(leftColoredImg)):
        new.append(list(leftColoredImg[i])+list(rightColoredImg[i]))

    # Show final image
    plt.imshow(new)
    plt.show()


improvedAgent(image)