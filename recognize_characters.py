import cv2
import numpy as np

from Area import Area
from constants import *
from preprocess import preprocess_image


def recognize_characters(original_image):
    gray_image, thresh_image = preprocess_image(original_image)
    possible_areas = find_area_with_chars(thresh_image)

    height, width = gray_image.shape
    image_board = np.zeros((height, width, 3), np.uint8)
    contours = []
    for area in possible_areas:
        contours.append(area.contour)
        cv2.drawContours(image_board, contours, -1, (255.0, 255.0, 255.0))
        cv2.imshow('a', image_board)
        cv2.waitKey()

    return []


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
