#!/usr/bin/python3
# Standard libraries
import numpy as np
from PIL import ExifTags
from PIL import Image


def get_property_from_file(image, property):
    # Get the corresponding tag for the selected property.
    if property == 'altitude':
        keyword = 'drone-dji:RelativeAltitude="'
    # Go over the image XMP data
    for segment, content in image.applist:
        marker, body = content.split('\x00', 1)
        # Work only with the relevant field of the XMP data
        if (segment == 'APP1'
                and marker == 'http://ns.adobe.com/xap/1.0/'):
            # When the keyword is found, extract the corresponding value.
            if keyword in body:
                altitude = body[body.index(keyword)+len(keyword) : 
                                body.index(keyword)+len(keyword)+6]
    return float(altitude)


def get_image_area(img_path):
    with Image.open(img_path) as image:
        altitude = get_property_from_file(image, "altitude")
        image_width = image.width
        image_height = image.height
    # Use the fov obtained from the camera specifications, and translate
    # to radians.
    # NOTE: Can't be used, as the degrees value don't seem accurate.
    # degrees = 84
    # fov = degrees * np.pi / 180
    # Get the FOV from the sensor geometry (21mm width, f=13.2mm).
    size = 0.021
    f = 0.0132
    fov = np.arctan((size/2)/f)*2
    # Find the number of meters per pixel, applying geometry formulas
    # with the camera parameters.
    image_real_width = 2 * np.tan(fov/2) * altitude
    pixel_width = image_real_width / image_width
    # Assuming that the pixel height and width are equal, find the image
    # total height.
    image_real_height = pixel_width * image_height
    image_area = image_real_height * image_real_width
    return image_area


def get_gsd(img_path, pumpkins):
    area = get_image_area(img_path)
    density = pumpkins / area
    return density


if __name__ == '__main__':
    density  = get_gsd('./photos/DJI_0237.JPG', pumpkins=2700)
    print("Assuming there are 10000 pumpkins in the field, the density"
          " is {0:.2f} pumpkins/m^2".format(density))
