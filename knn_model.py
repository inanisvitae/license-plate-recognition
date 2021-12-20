import cv2
import numpy as np


class KnnModel:
    """
    This class holds knn model and has api of classify for external use. Upon initializing, train() is evoked on knn
    model.
    """
    def __init__(self, train_labels, train_input):
        # https://stackoverflow.com/questions/49994760/opencv-assertion-failed-in-function-setdata
        # Needs to transform the dimension for knn_model.train()
        train_labels = train_labels.reshape(train_labels.size, 1)
        knn_model = cv2.ml.KNearest_create()
        knn_model.setDefaultK(1)
        # Needs to change to float32
        # https://medium.com/@sudhirjain01dec/optical-character-recognizer-using-knn-and-opencv-part2-57637649079c
        knn_model.train(np.array(train_input).astype('float32'), cv2.ml.ROW_SAMPLE,
                        np.array(train_labels).astype('float32'))
        self.knn_model = knn_model

    def classify(self, input_image):
        """
        findNearest evokes knn algorithm to classify input_image. Note that input_image is cropped out single image
        potentially containing character.
        :param input_image: Preprocessed image
        :return:
        """
        return self.knn_model.findNearest(input_image, k=1)
