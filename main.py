# 4.1.2 of OpenCV
import cv2

from knn_model import KnnModel
from preprocess import prepare_training_data, preprocess_image
from recognize_characters import recognize_characters


def main():
    """
    Main method for algorithm. In this part, preprocessing and training are done and reading original image into the
    recognition algorithm is also done here. Change the directory in cv2.imread() results in change of input image.
    :return:
    """
    [train_labels, train_input] = prepare_training_data()
    knn_model = KnnModel(train_labels, train_input)
    # Change path to jpeg to change images to read in
    original_image = cv2.imread('LicensePlatesImages/2.jpeg')
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
