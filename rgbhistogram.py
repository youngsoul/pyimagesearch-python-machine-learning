import cv2

"""
This histogram will be used to charac- terize the color of the flower petals, 
which is a good starting point for classifying the species of a flower
"""


class RGBHistogram:

    def __init__(self, bins):
        self.bins = bins

    def read_describe(self, imagePath):
        img = cv2.imread(imagePath)
        return self.describe(img)

    def describe(self, image, mask=None):
        hist = cv2.calcHist([image], [0, 1, 2],
                            mask, self.bins, [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)

        # return as a feature vector
        return hist.flatten()

