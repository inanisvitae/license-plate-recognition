import cv2


class Area:
    def __init__(self, contour):
        self.contour = contour
        self.bounding_rect = cv2.boundingRect(self.contour)
        [self.x, self.y, self.width, self.height] = self.bounding_rect

        self.size = self.width * self.height
        self.aspect_ratio = float(self.width) / float(self.height)

