import sys
import numpy as np
# 4.1.2 of OpenCV
import cv2

from knn_model import KnnModel
from preprocess import prepare_training_data, preprocess_image


def main():
    [train_labels, train_input] = prepare_training_data()
    knnModel = KnnModel(train_labels, train_input)
    original_image = cv2.imread('LicensePlatesImages/1.jpg')
    if original_image is None:
        print('\nError: image is not available \n')
        return

    gray_image, thresh_image = preprocess_image(original_image)
    thresh_image_copy = thresh_image.copy()
    contours, hierarchy = cv2.findContours(thresh_image_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# Source:
# https://medium.com/@sudhirjain01dec/optical-character-recognizer-using-knn-and-opencv-part2-57637649079c
