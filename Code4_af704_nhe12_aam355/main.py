import cv2
from basic import basicAgent
from improved import improvedAgent

def main():
    # Load in image
    image = cv2.imread("OgPic.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Call basic agent
    basicAgent(image)

    # Call improved agent
    improvedAgent(image)

if __name__ == "__main__":
    main()