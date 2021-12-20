import cv2
import numpy as np
from Area import Area
from constants import *
from preprocess import preprocess_image


def recognize_characters(knn_model, original_image):
    """
    Recognizes characters in original image. Preprocesses image, then iterate through all contours found in the image
    and check against rules, such as minimum area, aspect ratio, etc. Then iterate through all contours found and run
    knn classification algorithm on it.
    :param knn_model: Model trained with training set
    :param original_image: The raw image without preprocessing.
    :return: a string containing plate characters
    """
    gray_image, thresh_image = preprocess_image(original_image)
    possible_areas = find_area_with_chars(thresh_image)

    height, width = gray_image.shape
    image_board = np.zeros((height, width, 3), np.uint8)
    contours = []
    str_lst = []
    # TODO: Needs to group images together so that consecutive characters will be recognized together. Also need to
    #  remove contours that are within another contour. For example, needs to remove small triangle inside character '4'
    # Rearange characters with x, so license will be read from left to right
    possible_areas.sort(key=lambda item: item.x)
    for area in possible_areas:
        contours.append(area.contour)
        cv2.drawContours(image_board, contours, -1, (255.0, 255.0, 255.0))
        cv2.imshow('a', image_board)
        cv2.waitKey()
        cv2.rectangle(image_board, (area.x, area.y), (area.x + area.width, area.y + area.height), (0, 0, 255), 2)
        cv2.waitKey()
        potential_char_image = thresh_image[area.y:area.y+area.height, area.x:area.x+area.width]
        resized_potential_char_image = cv2.resize(potential_char_image, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        # Needs to reshape again for classification
        resized_potential_char_image = resized_potential_char_image.reshape((1, RESIZED_IMAGE_AREA))
        resized_potential_char_image = np.float32(resized_potential_char_image)
        retval, results, resp, dists = knn_model.classify(resized_potential_char_image)
        print(chr(int(results[0][0])))
        str_lst.append(chr(int(results[0][0])))
    return ''.join(str_lst)


def find_area_with_chars(thresh_image):
    possible_areas = []
    thresh_image_copy = thresh_image.copy()
    contours, hierarchy = cv2.findContours(thresh_image_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = Area(contour)
        # Should be more conditions, not just minimum area
        if area.size > MIN_PIXEL_AREA and area.width > MIN_PIXEL_WIDTH and area.height > MIN_PIXEL_HEIGHT \
                and MIN_ASPECT_RATIO < area.aspect_ratio < MAX_ASPECT_RATIO:
            possible_areas.append(area)
    return possible_areas


