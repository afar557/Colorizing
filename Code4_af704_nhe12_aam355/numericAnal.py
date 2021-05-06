from basic import *

def getAvgDist(imageName):
    new, original = basicAgent(imageName)
    avg = 0
    for i in range(len(new)):
        for j in range(len(new[0])):
            avg += euclidDist(new[i][j],original[i][j])
    avg /= (len(new)*len(new[0]))
    # return euclidDist(new, original)
    return avg

print(getAvgDist("OgPic.jpg"))