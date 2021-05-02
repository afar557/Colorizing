import cv2
import random
import numpy as np
import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
from copy import deepcopy
from statistics import mode
import sys

image = cv2.imread('palm2.webp')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def getRandWeights():
    out1W = np.random.rand(5,10)
	out2W = np.random.rand(3,6)
	return out1W, out2W

def sigmoid(dotProdArr):
    computedOut = 1/(1 + np.exp(-dotProdArr))
	return computedOut

def derivSigmoid(dotProdArr):
    computedOut = np.exp(-dotProdArr)/np.square(1 + np.exp(-dotProdArr))
	return computedOut

def getOutputFromModel(out1W, out2W, patch):
    patch = patch[]
    # out1 size is 6x1
    out1 = sigmoid(np.dot(out1W,np.transpose(patch)))
    np.insert(out1, 0, 1)
    # out2 size is 3x1
    out2 = sigmoid(np.dot(out2W,out1))
    return out1, out2

def lossFunc(out2, image, middlePixInd):
    x,y = middlePixInd
    loss = (out2[0] - image[x][y][0])**2 + (out2[1] - image[x][y][1])**2 + (out2[2] - image[x][y][2])**2
    return loss

def updateWeights(out1W, out2W, out1, out2, image, middlePixInd):
    x,y = middlePixInd
    middlePix = image[x][y]

    #Calculate weight1
    # derivLoss size is 3x3
    derivLoss = 2*np.diagflat(np.transpose(out2) - np.array([middlePix]))
    # sigPrime size is 3x1
    sigPrime = derivSigmoid(np.dot(out2W, out1))
    # dotProd size is 3x1
    dotProd = np.dot(derivLoss,sigPrime)
    # newWeight1 size is 3x6
    newWeight1 = np.dot(dotProd,np.transpose(out1))
    
    # Calculate weight2
    
    
    return newWeight1, newWeight2

def train(leftGrey, image, ):
    leftPatches = []
    for x in range(1,len(leftGrey)-1):
        for y in range(1,len(leftGrey[0])-1):
            leftPatch = np.array([1,leftGrey[x-1][y-1],leftGrey[x][y-1],leftGrey[x+1][y-1],[leftGrey[x-1][y],leftGrey[x][y],leftGrey[x+1][y]],leftGrey[x-1][y+1],leftGrey[x][y+1],leftGrey[x+1][y+1]])
            leftPatches.append([leftPatch,(x,y)])

    # get initial random weights
    out1W, out2W = getRandWeights()

    # pick  random patch and plug in patch into the model
    cntr = 0
    prevLoss = sys.maxsize
    while(true):
        x = random.randint(0,len(leftPatches)-1)
        out1, out2 = getOutputFromModel(out1W, out2W, leftPatches[x][0])
        loss = lossFunc(out2, image, leftPatches[x][1])
        cntr += 1
        if cntr == 1000:
            change = abs(prevLoss - loss) / prevLoss
            if change > .1:
                prevLoss = loss
                cntr = 0
                continue
            else:
                break
        

        


def recolorRight():

def improvedAgent():