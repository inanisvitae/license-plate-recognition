import math

import cv2


class Area:
    """
    This class simply stores information regarding position, size of a contour
    """
    def __init__(self, contour):
        self.contour = contour
        self.bounding_rect = cv2.boundingRect(self.contour)
        [self.x, self.y, self.width, self.height] = self.bounding_rect
        self.size = self.width * self.height
        self.aspect_ratio = float(self.width) / float(self.height)

        self.diagonal = math.sqrt((self.width ** 2) + (self.height ** 2))

        self.center_x = (self.x + self.y + self.width) / 2
        self.center_y = (self.y + self.y + self.height) / 2
