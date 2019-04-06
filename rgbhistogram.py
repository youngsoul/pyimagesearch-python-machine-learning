import cv2
import numpy as np

"""
This histogram will be used to charac- terize the color of the flower petals, 
which is a good starting point for classifying the species of a flower
"""


class RGBHistogram:

    def __init__(self, bins):
        self.bins = bins

    def get_features(self, imagePath):
        img = cv2.imread(imagePath)
        features = []
        if img is not None:
            features.extend(self.extract_color_stats(img))
            features.extend(self.describe(img).tolist())

        return features

    def extract_color_stats(self, image):
        # split the input image into its respective RGB color channels
        # and then create a feature vector with 6 values: the mean and
        # standard deviation for each of the 3 channels, respectively
        (B,G, R) = cv2.split(image)
        stats = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
                    np.std(G), np.std(B)]

        # perform a min/max scaling on the mean and standard deviation since we are
        # normalizing the histogram bins as well.
        np_stats = np.array(stats)
        normal_np_stats = np.interp(np_stats, (np_stats.min(), np_stats.max()), (0, +1))

        # return our set of features
        return normal_np_stats

    def describe(self, image, mask=None):
        hist = cv2.calcHist([image], [0, 1, 2],
                            mask, self.bins, [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)

        # return as a feature vector
        return hist.flatten()

