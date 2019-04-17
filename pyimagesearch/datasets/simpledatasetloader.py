import numpy as np
import cv2
import os


class SimpleDatasetLoader:

    def __init__(self, preprocessors=[]):
        # store impage preprocessor
        self.preprocessors = preprocessors

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming
            # that our path has the following format
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # check to see if our processors exist
            if self.preprocessors:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our processed image as a "feature vector"
            # by updating hte data list followed by the labels
            data.append(image)
            labels.append(label)

            # show an update every 'verbose' images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print(f"[INFO] process {i + 1}/{len(imagePaths)}")

        if verbose > 0:
            print(f"[INFO] loaded {len(data)} rows")
        return np.array(data), np.array(labels)
