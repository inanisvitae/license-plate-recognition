import sys
import numpy as np
# 4.1.2 of OpenCV
import cv2

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

RESIZED_WIDTH = 20
RESIZED_HEIGHT = 30


def preprocess_image(input_image):
    # Make image gray to remove colors
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    blurry_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresh_image = cv2.adaptiveThreshold(
        blurry_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    return gray_image, thresh_image


def prepare_training_data():
    training_image = cv2.imread('training_chars.png')

    if training_image is None:
        print('Image not available\n')
        return
    gray_image, thresh_image = preprocess_images(training_image)
    cv2.imshow('Image Thresh', thresh_image)
    cv2.waitKey()
    copy_thresh_image = thresh_image.copy()
    # https://docs.opencv.org/4.1.2/d4/d73/tutorial_py_contours_begin.html
    contours, hierarchy = cv2.findContours(copy_thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    train_input = np.empty((0, RESIZED_WIDTH * RESIZED_HEIGHT))

    labels = []
    valid_chars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                    ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z'),
                     ord('a'), ord('b'), ord('c'), ord('d'), ord('e'), ord('f'), ord('g'), ord('h'), ord('i'), ord('j'),
                     ord('k'), ord('l'), ord('m'), ord('n'), ord('o'), ord('p'), ord('q'), ord('r'), ord('s'), ord('t'),
                     ord('u'), ord('v'), ord('w'), ord('x'), ord('y'), ord('z')]
    # keys_pressed = []
    keys_pressed = np.loadtxt("keys_pressed.txt", np.float32)
    count = 0
    for contour in contours:
        print(cv2.contourArea(contour))
        if 100 < cv2.contourArea(contour) < 1000:
            [x, y, w, h] = cv2.boundingRect(contour)
            cv2.rectangle(training_image, (x, y), (x + w, y + h), (0, 0, 0), 2)
            individual_area = thresh_image[y:y + h, x:x + w]
            individual_area_resized = cv2.resize(thresh_image[y:y + h, x:x + w], (RESIZED_WIDTH, RESIZED_HEIGHT))
            cv2.imshow("individual_area", individual_area)
            cv2.imshow("individual_area_resized", individual_area_resized)
            cv2.imshow("training_numbers.png", training_image)
            # key_pressed = cv2.waitKey(0)
            key_pressed = keys_pressed[count]
            # keys_pressed.append(key_pressed)
            if key_pressed in valid_chars:
                labels.append(key_pressed)
                train_input = np.append(train_input, individual_area_resized.reshape((1, RESIZED_WIDTH * RESIZED_HEIGHT)), 0)
            count += 1
    train_labels_mat = np.array(labels, np.float32)
    train_labels = train_labels_mat.reshape((train_labels_mat.size, 1))

    print("\n\ntraining complete !!\n")

    # np.savetxt("train_labels.txt", train_labels)
    # np.savetxt("train.txt", train_input)
    # np.savetxt('keys_pressed.txt', keys_pressed)
    cv2.destroyAllWindows()
    return train_labels, train_input
