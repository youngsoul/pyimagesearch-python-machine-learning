from sklearn.externals import joblib
from PIL import Image
import numpy as np
from rgbhistogram import RGBHistogram

rgbHisto = RGBHistogram([8, 8, 8])


def extract_color_stats(image):
    # split the input image into its respective RGB color channels
    # and then create a feature vector with 6 values: the mean and
    # standard deviation for each of the 3 channels, respectively
    (R, G, B) = image.split()
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
                np.std(G), np.std(B)]

    # return our set of features
    return features


def load_targets():
    targets = []
    with open('./3scenes_scene_labels.txt', 'r') as f:
        line = f.readline().strip()
        while line:
            targets.append(line)
            line = f.readline().strip()

    return targets


def predict_scene(imagePath):
    print(f'Predict for image: {imagePath}')
    model = joblib.load('3scenes_image_classify_scikit_model.sav')
    features = rgbHisto.get_features(imagePath)
    prediction = model.predict([features])

    print(f'Prediction: {prediction}')
    print(f'Targets: {load_targets()}')



if __name__ == '__main__':
    test_image = './3scenes_holdout/highway/highway_art820.jpg'
    predict_scene(test_image)
