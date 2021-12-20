import sys
import numpy as np
# 4.1.2 of OpenCV
import cv2

from knn_model import KnnModel
from preprocess import prepare_training_data, preprocess_image
from recognize_characters import recognize_characters


def main():
    [train_labels, train_input] = prepare_training_data()
    knn_model = KnnModel(train_labels, train_input)
    original_image = cv2.imread('LicensePlatesImages/1.jpeg')
    if original_image is None:
        print('\nError: image is not available \n')
        return

    characters_result = recognize_characters(knn_model, original_image)
    print('Recognized plate from image = ' + characters_result.upper())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# Source:
# https://medium.com/@sudhirjain01dec/optical-character-recognizer-using-knn-and-opencv-part2-57637649079c
