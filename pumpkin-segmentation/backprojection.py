#!/usr/bin/python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

if __name__ == "__main__":
    
    #loading images
    path_to_original_img = "photos/DJI_0237.JPG"
    path_to_manually_marked_img = "photos/DJI_0237_copy_better.png"
    
    img_original = cv2.imread(path_to_original_img)
    img_marked = cv2.imread(path_to_manually_marked_img)
    
    #get masked image
    min_bgr_thres = np.array([0, 0, 240])
    max_bgr_thres = np.array([0, 0, 255])
    mask = cv2.inRange(img_marked,min_bgr_thres,max_bgr_thres)
    img_masked = cv2.bitwise_and(img_original, img_original, mask = mask)
    img_masked_hsv = cv2.cvtColor(img_masked, cv2.COLOR_BGR2HSV)   
    
    #split image into channels
    #RGB
    b = img_masked[:,:,2]
    b = b.reshape(b.shape[0]*b.shape[1])
    g = img_masked[:,:,1]
    g = g.reshape(g.shape[0]*g.shape[1])     
    r = img_masked[:,:,0]    
    r = r.reshape(r.shape[0]*r.shape[1])
    
    #HSV
    h = img_masked_hsv[:,:,0]
    h = h.reshape(h.shape[0]*h.shape[1])
    s = img_masked_hsv[:,:,1]
    s = s.reshape(s.shape[0]*s.shape[1])
    
    #plotting histograms
    #RGB
    nbins = 50
    plt.figure("RGB backprojection")
    plt.grid(True)
    plt.subplot(1, 3, 1)
    plt.hist2d(r,g,bins=nbins,norm=LogNorm())
    plt.xlabel('Red')
    plt.ylabel('Green')
    plt.xlim([0,255])
    plt.ylim([0,255])
    plt.subplot(1, 3, 2)
    plt.title("Variance of pumpkin color in RGB space")
    plt.hist2d(r,b,bins=nbins,norm=LogNorm())
    plt.xlabel('Red')
    plt.ylabel('Blue')
    plt.subplot(1, 3, 3)
    plt.hist2d(b,g,bins=nbins,norm=LogNorm())
    plt.xlabel('Blue')
    plt.ylabel('Green')
    plt.colorbar()
    plt.show()
    
    #HSV
    plt.figure("HSV backprojection")
    plt.title("Variance of pumpkin color in HSV space")
    plt.hist2d(h,s,bins=nbins,norm=LogNorm())
    plt.xlabel("Hue")
    plt.ylabel("Saturation")
    plt.colorbar()
    plt.show()
    

    