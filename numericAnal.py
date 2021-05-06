from basic import basicAgent
import numpy as np

def euclidDist(a , b):
    dist = np.linalg.norm(a-b)
    return(dist)

def getAvgDist(imageName):
    new, original = basicAgent(imageName)
    avg = 0
    for i in range(len(new)):
        for j in range(len(new[0])):
            avg += euclidDist(new[i][j],original[i][j])
    avg /= (len(new)*len(new[0]))
    # return euclidDist(new, original)
    return avg

print(getAvgDist("aiai.jpg"))



# print(euclidDist( np.array([255,255,255]), np.array([0,0,0]) ) )