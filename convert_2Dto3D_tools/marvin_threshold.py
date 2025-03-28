# marvin_threshold.py

import cv2
import numpy as np
from pathlib import Path

"""Given an image, this script will threshold the image to detect the background color and create a mask for the foreground.
BGR values based on original maxi marvin code"""

def threshold_image(file_name, file_name_result):
    # Load the image
    image = cv2.imread(str(file_name))

    # Define background color range (R=0<>227, G=0<>242, B=0<>122)
    lower_bound = np.array([0, 0, 0])     # Lower bounds for B, G, R
    upper_bound = np.array([122, 242, 227])  # Upper bounds for B, G, R

    # Create a mask where the background color is detected
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Invert mask to get the foreground (everything that's not background)
    foreground_mask = cv2.bitwise_not(mask)

    # Apply the mask to the original image to get the foreground
    result = cv2.bitwise_and(image, image, mask=foreground_mask)

    # Convert the result to a binary image (optional for thresholding)
    _, binary_result = cv2.threshold(foreground_mask, 1, 255, cv2.THRESH_BINARY)

    # result_orig = cv2.imread(str(file_name_result))
    # # result_orig[result_orig!=0]=255
    # result_orig_bin = np.zeros(result_orig.shape[:2])
    # result_orig_bin[np.all(result_orig==0,axis=2)]=255

    # Optionally display images
    # cv2.imshow('Original Image', image)
    cv2.imshow('Binary Threshold Image', binary_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    folder = Path('example_data')
    plant_nr = "Harvest_02_PotNr_27"
    file_name = folder  / "images" / plant_nr / "21-92-002226018-cam_00.png"
    file_name_result = folder / "inference" / plant_nr / "21-92-002226018-preseg-cam_00.png"

    threshold_image(file_name, file_name_result)


