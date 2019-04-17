from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:

    def __init__(self, dataFormat=None, flatten_array=False):
        # store image data format
        self.dataFormat = dataFormat
        self.flatten_array = flatten_array

    def preprocess(self, image):
        # apply the Keras utility function that correctly rearranges the dimensions of the image
        i = img_to_array(image, data_format=self.dataFormat)
        if self.flatten_array:
            i = i.flatten()

        return i