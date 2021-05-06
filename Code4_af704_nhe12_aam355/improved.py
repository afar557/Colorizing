import cv2
import random
import numpy as np
import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
from copy import deepcopy
from statistics import mode
import sys

# Main function that runs the improved agent algorithm
def improvedAgent(image):
    # Convert image to grey scale
    greyImg = greyScaleImg(deepcopy(image))

    # Spilt the grey image into the right and left side
    half = int(len(image[0])/2)

    leftGrey = greyImg[:,:half]
    rightGrey = greyImg[:,half:]

    # Train the model using the left side of the image
    out1W, out2W = train(deepcopy(leftGrey), deepcopy(image))

    # Recolor the right side of the image using the weights from the training
    rightColoredImg = recolorRight(deepcopy(image), deepcopy(rightGrey), out1W, out2W)

    # Combine the recolored right and original left image
    leftColoredImg = image[:,:half]
    new = []
    for i in range(0, len(leftColoredImg)):
        new.append(list(leftColoredImg[i])+list(rightColoredImg[i]))

    # Show final image
    plt.imshow(new)
    plt.show()

# Function that trains the model using the left side of the grey image
def train(leftGrey, image):
    # Get the left patches from the left side of the image
    leftPatches = []
    for x in range(1,len(leftGrey)-1):
        for y in range(1,len(leftGrey[0])-1):
            leftPatch = np.array([[1,leftGrey[x-1][y-1],leftGrey[x][y-1],leftGrey[x+1][y-1],leftGrey[x-1][y],leftGrey[x][y],leftGrey[x+1][y],leftGrey[x-1][y+1],leftGrey[x][y+1],leftGrey[x+1][y+1]]])
            leftPatches.append([leftPatch/255,(x,y)])

    # Get initial random weights
    out1W, out2W = getRandWeights()

    # Pick  random patch and plug in patch into the model
    cntr = 0
    avgPrevLoss = sys.maxsize
    lossArr = []
    countArr = []
    sumLoss = 0
    alpha = 0.01
    while(True):
        # Pick a random patch from the left patches
        x = random.randint(0,len(leftPatches)-1)

        # Get the outputs by plugging in the weights into the equations
        out1, out2 = getOutputFromModel(out1W, out2W, leftPatches[x][0])

        # Calculate the loss function using the output
        loss = lossFunc(out2, deepcopy(image), leftPatches[x][1])
        countArr.append(cntr)
        lossArr.append(loss)
        # print("loss is :", loss)

        # Calculate the new
        newWeight1, newWeight2 = updateWeights(out1W, out2W, out1, out2, deepcopy(image), leftPatches[x][1], leftPatches[x][0])

        # Update our current weights using the new weights we got using the derivtive of the loss function
        out1W -= (alpha*newWeight1)
        out2W -= (alpha*newWeight2)

        cntr += 1
        
        # One we run 1000 pixels, check the loss function to see if it has a 5% decrease and if so break, otherwise do 1000 more iterations
        sumLoss += loss

        print("current count: ", cntr)
        if cntr == 100000:
            # plt.plot(countArr, lossArr)
            # plt.show()
            # lossArr=[]
            # countArr=[]
            avgLoss = sumLoss/1000
            change = abs((avgLoss - avgPrevLoss) / avgPrevLoss)
            if change > .01:
                avgPrevLoss = avgLoss
                cntr = 0
                alpha=alpha/2
                continue
            else:
                break
    return out1W, out2W

# Function that produces random weights for the hidden and output layer
def getRandWeights():
    out1W = np.random.rand(6,10)
    out2W = np.random.rand(3,6)
    return out1W, out2W

# Function that uses the model to get the output(out2) and hidden layer(out2)
def getOutputFromModel(out1W, out2W, patch):
    # Find out1/hidden layer by getting the dot product of the patch/input and the weights for out1
    # 6x10 * 10x1 = 6x1
    out1 = sigmoid(np.dot(out1W,np.transpose(patch)))

    # Find out2/output layer by getting the dot product of out1 and the weights of out2
    # 3x6 * 6x1 = 3x1
    out2 = sigmoid(np.dot(out2W,out1))
    return out1, out2

# Function that calculates the sigmoid as per example network notes
def sigmoid(dotProdArr):
    computedOut = 1/(1 + np.exp(-dotProdArr))
    return computedOut

# Function that calculates the loss
def lossFunc(out2, image, middlePixInd):
    x,y = middlePixInd
    middlePix = image[x][y]/255
    loss = (out2[0] - middlePix[0])**2 + (out2[1] - middlePix[1])**2 + (out2[2] - middlePix[2])**2
    return loss

# Function that excutes the derivative of the loss function
def updateWeights(out1W, out2W, out1, out2, image, middlePixInd, patch):
    x,y = middlePixInd
    middlePix = image[x][y]/255

    # Calculate weight2 for out2W
    # 1x3 - 1x3 = 1x3
    # diagflat(1x3) = 3x3
    derivLoss = 2*np.diagflat(np.transpose(out2) - np.array([middlePix]))

    # 3x6 * 6x1 = 3x1
    sigPrime = derivSigmoid(np.dot(out2W, out1))

    # 3x3 * 3x1 = 3x1
    dotProd = np.dot(derivLoss,sigPrime)
    
    # 3x1 * 1x6 = 3x6 newWeight2 size is 3x6
    newWeight2 = np.dot(dotProd,np.transpose(out1))
    
    # Calculate weight1 for out1W

    # Calculate the deriv term by doing a sum
    derivLoss2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(3):
        # 1x1 - 1x1 = 1x1
        derivLoss3 = 2*(out2[i] - middlePix[i])

        # 1x6 * 6x1 = 1x1
        sigPrime2 = derivSigmoid(np.dot(out2W[i],out1))

        # 1x1 * 1x1 * 1x6 = 1x6
        prod = derivLoss3 * sigPrime2 * out2W[i]

        # 1x6
        derivLoss2 += (prod)

    # diagflat(1x6) = 6x6
    derivLoss2 = np.diagflat(derivLoss2)

    # 6x10 * 10x1 = 6x1
    sigPrime3 = derivSigmoid(np.dot(out1W, np.transpose(patch)))

    # 6x1 * 1x10 = 6x10
    dotProd2 = np.dot(sigPrime3,patch)

    # 6x6 * 6x10 = 6x10
    newWeight1 = np.dot(derivLoss2,dotProd2)

    return newWeight1, newWeight2

# Function that calculates the sigmoid prime as per example network notes
def derivSigmoid(dotProdArr):
    computedOut = np.exp(-dotProdArr)/np.square(1 + np.exp(-dotProdArr))
    return computedOut

# Function that recolors the right side of the grey image using the model we trained  
def recolorRight(image, rightGrey, out1W, out2W):
    half = int(len(image[0])/2)
    rightToColor = image[:,half:]

    for i in range(1,len(rightGrey)-1):
        for j in range(1,len(rightGrey[0])-1):
            # get 3x3 gray patch from right side
            rightPatch = np.array([[1,rightGrey[i-1][j-1],rightGrey[i][j-1],rightGrey[i+1][j-1],rightGrey[i-1][j],rightGrey[i][j],rightGrey[i+1][j],rightGrey[i-1][j+1],rightGrey[i][j+1],rightGrey[i+1][j+1]]])
            out2 = getOutputFromModel(out1W, out2W, rightPatch/255)[1]
            newR, newG, newB = out2

            rightToColor[i][j][0] = newR*255
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

improvedAgent("aiai.jpg")