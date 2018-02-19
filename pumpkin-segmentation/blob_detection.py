# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    #Read the color based segemented image
    imageBGR = cv2.imread("./photos/DJI_0237.JPG")
    imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

    binaryImage = 255*cv2.imread("photos/result.png", 0)

    #Morphological filtering to seprate joint groups and get rid of the noise
    kernel = np.ones((3, 3), np.uint8)
    erodedImage = cv2.erode(binaryImage, kernel, iterations = 1)

    #Find contours in the image
    im2, contours, hierarchy = cv2.findContours(erodedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   
   
    #Filter detected contours, save the remaing in filtered_contours array
    filtered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        
        if area > 50:
            print("big enough!")
            filtered_contours.append(contour)            
            
    #Draw contours of groups expected to be pumpkins
    cv2.drawContours(imageRGB, filtered_contours, -1, (0, 0, 255), 3)
    
    #Reduction of size of the contours in filtering process
    print(len(contours))  
    print(len(filtered_contours))  

    #Print the image
    plt.figure("name of the figure")
    plt.imshow(imageRGB)
    plt.show()
    
    #Write the image
    cv2.imwrite("./photos/result.png", imageRGB)

