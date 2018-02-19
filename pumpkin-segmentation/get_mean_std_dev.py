#!/usr/bin/python3
import cv2
import matplotlib
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def print_image(image_orig, image_result, name):
    plt.figure(name)
    plot_original = plt.subplot(2, 1, 1)
    plot_original.imshow(image_orig)
    plot = plt.subplot(2, 1, 2,
                       sharex=plot_original, sharey=plot_original)
    plt.imshow(image_result)

def load_pictures(path):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img_rgb, img_hsv

def get_mean_st_dev(image_training, image_orig):
    image_drawed_rgb, image_drawed_hsv = load_pictures(image_training)
    image_original_rgb, image_original_hsv = load_pictures(image_orig)
    # red color is used to draw on pumpkins in test image
    # thresholding in red channel
    min_rgb_thres = np.array([240, 0, 0])
    max_rgb_thres = np.array([255, 0, 0])
    color_chan = 0 # red (RGB)
    mask_image = get_masked_image(min_rgb_thres, max_rgb_thres,
                                  image_drawed_rgb, image_original_rgb,
                                  color_chan)

    mean_rgb, std_dev_rgb = cv2.meanStdDev(image_original_rgb, mask=mask_image)
    mean_hsv, std_dev_hsv = cv2.meanStdDev(image_original_hsv, mask=mask_image)

    # used for visual test
    compare_result_image_with_original(mean_rgb, std_dev_rgb,
                                       image_original_rgb, "RGB")
    compare_result_image_with_original(mean_hsv, std_dev_hsv,
                                       image_original_hsv, "HSV")
    plt.show()
    return mean_rgb, mean_hsv, std_dev_rgb, std_dev_hsv

## used to show working method by removing all but the pixels
## with valid threshold values
def compare_result_image_with_original(mean, std_dev, original_image, name):
    min_val = np.array(mean - std_dev)
    max_val = np.array(mean + std_dev)
    mask = cv2.inRange(original_image,min_val,max_val)
    result = cv2.bitwise_and(original_image, original_image,
                                 mask=mask)
    print_image(original_image, result,
                "Original image and result image - " + name)

# threshold a specified color and return the original image
# showing only the pixels within this range
def get_masked_image(min_thres, max_thres, image_drawed,
                     image_original, color_channel):
    mask_rgb = cv2.inRange(image_drawed,min_thres,max_thres)
    # this will return only the pixels which are drawed manually
    result_rgb = cv2.bitwise_and(image_original, image_original,
                                 mask=mask_rgb)
    return mask_rgb

if __name__ == "__main__":
    path_to_training_image = "photos/DJI_0237_copy_better.png"
    path_to_original_image = "photos/DJI_0237.JPG"
    mean_rgb, mean_hsv, std_dev_rgb, std_dev_hsv = get_mean_st_dev(path_to_training_image,
                                    path_to_original_image)
    print "Mean RGB\n" + str(mean_rgb)
    print "StdDev RGB\n" + str(std_dev_rgb)

    print "\n\nMean HSV\n" + str(mean_hsv)
    print "StdDev HSV\n" + str(std_dev_hsv)
