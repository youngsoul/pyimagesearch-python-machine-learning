from sklearn.externals import joblib
from PIL import Image
import numpy as np

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
    with open('./scene_labels.txt', 'r') as f:
        line = f.readline().strip()
        while line:
            targets.append(line)
            line = f.readline().strip()

    return targets


def predict_scene(imagePath):
    model = joblib.load('image_classify_scikit_model.sav')

    image = Image.open(imagePath)
    the_labels = targets
    features = extract_color_stats(image)

    prediction = model.predict([features])

    return prediction

if __name__ == '__main__':
    test_image = './3scenes_holdout/highway/highway_art820.jpg'
    targets = load_targets()
    print(targets)
    pred = predict_scene(test_image)
    print(test_image)
    print(pred)
    print(targets)

