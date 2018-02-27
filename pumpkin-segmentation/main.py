#!/usr/bin/python3
# Standard libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
# Local libraries
import binarization
import blob_detection
import feature_extraction
import get_mean_std_dev


def main(save_image=False, show_result=False):
    """Main routine."""
    # Read photo
    training_image_path = "./photos/DJI_0237_copy_better.png"
    image_path = "./photos/DJI_0237.JPG"
    image = cv2.imread(image_path)
    # Get the statistics of the pumpkins.
    stats = get_mean_std_dev.get_mean_st_dev(training_image_path, image_path)
    h_thresholds = get_mean_std_dev.get_thresholds(stats[1][0], stats[3][0])
    s_thresholds = get_mean_std_dev.get_thresholds(stats[1][1], stats[3][1])
    v_thresholds = get_mean_std_dev.get_thresholds(stats[1][2], stats[3][2])
    # Binarize the image.
    binarized = binarization.hsv_binarization(image, h_thresholds, s_thresholds,
                                              v_thresholds)
    if save_image:
        cv2.imwrite("./photos/result.png", 255*binarized)
    # Detect the blobs on the binarized image
    n_pumpkins, blobs = blob_detection.blob_detection(image, binarized*255)
    if show_result:
        print ("showing")
        plt.imshow(blobs)
        plt.show()
    # Extract the density and gsd
    density, gsd = feature_extraction.get_density(image_path, n_pumpkins)
    print("The Ground sample distance (GSD) is {0:.4f} m/px".format(gsd))
    print("Detected {} pumpkins".format(n_pumpkins))
    print("The density is {0:.2f} pumpkins/m^2".format(density))
    return


if __name__ == "__main__":
    save_image = False
    show_result = True
    main(save_image, show_result)
