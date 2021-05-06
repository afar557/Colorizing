from basic import basicAgent
from improved import improvedAgent
import numpy as np

def euclidDist(a , b):
    dist = np.linalg.norm(a-b)
    return(dist)

def getAvgDist(image):
    # new, original = basicAgent(image)
    new, original = improvedAgent(image)
    avg = 0
    for i in range(len(new)):
        for j in range(len(new[0])):
            avg += euclidDist(new[i][j],original[i][j])
    avg /= (len(new)*len(new[0]))
    # return euclidDist(new, original)
    return avg
