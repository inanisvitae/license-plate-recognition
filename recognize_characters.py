import math

import cv2
import numpy as np

from Area import Area
from constants import *
from preprocess import preprocess_image


def group_characters(possible_areas):
    """
    Groups characters together from a list of areas. Calculates distance and angles between different areas
    :param possible_areas: a list of areas, containing contour in each area object
    :return: a list of groups that contain areas
    """
    list_of_groups_of_characters = []
    for pa in possible_areas:
        group_of_characters = group_list_characters(pa, possible_areas)
        group_of_characters.append(pa)

        if len(group_of_characters) < MIN_NUMBER_OF_MATCHING_CHARS:
            continue
        # Obtains a list of groups of possible characters
        list_of_groups_of_characters.append(group_of_characters)
        list_of_characters_removed = list(set(possible_areas) - set(group_of_characters))
        recursive_list = group_characters(list_of_characters_removed)
        for recursive_pa in recursive_list:
            list_of_groups_of_characters.append(recursive_pa)
        break

    return list_of_groups_of_characters


def compute_euclidean_distance_between_areas(first_area, second_area):
    """
    Calculates euclidean distance between centers of two areas
    :param first_area: First area to compute
    :param second_area: Second area to compute
    :return: Numerical value of distance
    """
    x = abs(first_area.x - second_area.x)
    y = abs(first_area.y - second_area.y)
    return math.sqrt((x ** 2) + (y ** 2))


def remove_overlapped_characters(possible_areas):
    list_of_characters_removed = list(possible_areas)
    for pa1 in possible_areas:
        for pa2 in possible_areas:
            if pa1 is pa2:
                continue
            if compute_euclidean_distance_between_areas(pa1, pa2) < (pa1.diagonal * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                if pa1 in list_of_characters_removed:
                    list_of_characters_removed.remove(pa1)
            else:
                if pa2 in list_of_characters_removed:
                    list_of_characters_removed.remove(pa2)
    return list_of_characters_removed


def group_list_characters(possible_area, possible_areas):
    """
    Checks whether two areas should be grouped together according to some rules. The rules are as follows: Change in area
    shouldn't be smaller than MAX_CHANGE_IN_AREA,
    :param possible_area: Area to be grouped
    :param possible_areas: the list of areas
    :return:
    """
    list_of_grouped_area = []
    for pa in possible_areas:
        if possible_area is pa:
            continue
        distance_between_areas = compute_euclidean_distance_between_areas(pa, possible_area)
        change_in_areas = float(abs(pa.size - possible_area.size) / float(possible_area.size))
        change_in_width = float(abs(pa.width - possible_area.width) / float(possible_area.width))
        change_in_height = float(abs(pa.height - possible_area.height) / float(possible_area.height))
        change_in_y_distance = abs(pa.center_y - possible_area.center_y)
        average_height = abs(pa.height + possible_area.height) / 2.0
        if (distance_between_areas < (pa.diagonal * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
                change_in_areas < MAX_CHANGE_IN_AREA and
                change_in_width < MAX_CHANGE_IN_WIDTH and
                change_in_height < MAX_CHANGE_IN_HEIGHT and
                # Change in distance in y axis shouldn't be smaller than half of height of the character
                change_in_y_distance < average_height):
            list_of_grouped_area.append(pa)
    return list_of_grouped_area


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
    list_of_groups_of_characters = group_characters(possible_areas)
    for list_of_group in list_of_groups_of_characters:
        list_of_group.sort(key=lambda matching_char: matching_char.center_x)
    list_of_groups_of_characters_tmp = max(list_of_groups_of_characters, key=len)
    for area in list_of_groups_of_characters_tmp:
        contours.append(area.contour)
        cv2.drawContours(image_board, contours, -1, (255.0, 255.0, 255.0))
        cv2.imshow('a', image_board)
        cv2.waitKey()
        cv2.rectangle(image_board, (area.x, area.y), (area.x + area.width, area.y + area.height), (0, 0, 255), 2)
        cv2.waitKey()
        potential_char_image = thresh_image[area.y:area.y + area.height, area.x:area.x + area.width]
        resized_potential_char_image = cv2.resize(potential_char_image, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        # Needs to reshape again for classification
        resized_potential_char_image = resized_potential_char_image.reshape((1, RESIZED_IMAGE_AREA))
        resized_potential_char_image = np.float32(resized_potential_char_image)
        retval, results, resp, dists = knn_model.classify(resized_potential_char_image)
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
