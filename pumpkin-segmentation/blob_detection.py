# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


def estimatePumpkins(filtered_contours, total_area):
    #calculating mean area for a pumpkin
    mean_area = total_area/len(filtered_contours)
    
    #estimating number of pumpkins based on mean area
    estimated_pumpkins_nr = 0
    
    for contour in filtered_contours:
        area = cv2.contourArea(contour)
        
        if area > mean_area:
            estimated_pumpkins_nr = estimated_pumpkins_nr + int(round(area/mean_area))
        else:
            estimated_pumpkins_nr = estimated_pumpkins_nr + 1
            
    return estimated_pumpkins_nr

def blob_detection(imgRGB, imgBinary):
    #Morphological filtering to seprate joint groups and get rid of the noise
    kernel = np.ones((3, 3), np.uint8)
    imgEroded = cv2.erode(imgBinary, kernel, iterations = 1)

    #Find contours in the image
    im2, contours, hierarchy = cv2.findContours(imgEroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   
   
    #Filter detected contours, save the remaing in filtered_contours array
    filtered_contours = []
    total_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > 50:
            total_area = total_area + area
            filtered_contours.append(contour)            
    
    pumpkins_estimated = estimatePumpkins(filtered_contours, total_area)
 
    #Draw contours of groups expected to be pumpkins
    imgBlobDetected = imgRGB.copy()
    cv2.drawContours(imgRGB, filtered_contours, -1, (0, 0, 255), 3)
    
    #Priting out: number of contours and pumpkins
    # print("Number of contours detected: %d" % len(contours))
    # print("Number of pumpkins detected: %d" % len(filtered_contours))
    # print("Number of pumpkins estimated based on mean area: %d" % pumpkins_estimated)

    #Print the image
    #plt.figure("name of the figure")
    #plt.imshow(imgRGB)
    #plt.show()
    
    #Write the image
    cv2.imwrite("./photos/blob_detection.png", imgRGB)    
    
    return (pumpkins_estimated, imgBlobDetected)

if __name__ == "__main__":
    
    #Read original photo
    imageBGR = cv2.imread("./photos/DJI_0237.JPG")
    imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

    #Read color based segemented image
    binaryImage = 255*cv2.imread("photos/result.png", 0)
    
    blob_detection(imageRGB, binaryImage)
    
    

