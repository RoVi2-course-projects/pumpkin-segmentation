#!/usr/bin/python3
import cv2
import matplotlib
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def print_image(image, name):
    plt.figure(name)
    plt.imshow(image)

def load_pictures_rgb(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_mean_st_dev(image_drawed, image_original):
    image_drawed = load_pictures_rgb("DJI_0237_copy.JPG")
    image_original = load_pictures_rgb("DJI_0237.JPG")
    # red color is used to draw on pumpkins in test image
    # thresholding in red channel
    min_rgb_thres = np.array([240, 0, 0])
    max_rgb_thres = np.array([255, 0, 0])
    color_chan = 0 # red (RGB)
    mask_image = get_masked_image(min_rgb_thres, max_rgb_thres,
                                  image_drawed, image_original, color_chan)

    mean, std_dev = cv2.meanStdDev(image_original, mask=mask_image)
    # print mean
    # print std_dev

    # used to show workign method by removing all but the pixels
    # with valid threshold values
    min_rgb = np.array(mean - std_dev)
    max_rgb = np.array(mean + std_dev)
    mask_rgb = cv2.inRange(image_original,min_rgb,max_rgb)
    result_rgb = cv2.bitwise_and(image_original, image_original, mask=mask_rgb)
    print_image(image_original, "Original image")
    print_image(result_rgb, "Result for part 1")
    plt.show()
    return mean, std_dev

# threshold a specified color and return the original image
# showing only the pixels within this range
def get_masked_image(min_thres, max_thres, image_drawed,
                     image_original, color_channel):
    mask_rgb = cv2.inRange(image_drawed,min_thres,max_thres)
    # this will return only the pixels which are drawed manually
    result_rgb = cv2.bitwise_and(image_original, image_original, mask=mask_rgb)
    return mask_rgb

if __name__ == "__main__":
    path_to_training_image = "DJI_0237_copy.JPG"
    path_to_original_image = "DJI_0237.JPG"
    mean, std_dev = get_mean_st_dev(path_to_training_image,
                                    path_to_original_image)
