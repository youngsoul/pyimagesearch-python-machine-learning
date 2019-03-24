from sklearn.externals import joblib
from PIL import Image
import numpy as np
from rgbhistogram import RGBHistogram

rgbHisto = RGBHistogram([8, 8, 8])


def load_targets():
    targets = []
    with open('./animals_scene_labels.txt', 'r') as f:
        line = f.readline().strip()
        while line:
            targets.append(line)
            line = f.readline().strip()

    return targets


def predict_scene(model, imagePath):
    print(f'Predict for image: {imagePath}')
    features = rgbHisto.get_features(imagePath)
    prediction = model.predict([features])

    print(f'Prediction: {prediction}')
    print(f'Targets: {load_targets()}')



if __name__ == '__main__':
    test_images = [
        './animal_holdout/cats/cats_00843.jpg',
        './animal_holdout/cats/cats_00997.jpg',
        './animal_holdout/dogs/dogs_00102.jpg',
        './animal_holdout/dogs/dogs_00163.jpg',
        './animal_holdout/pandas/panda_00050.jpg',
        './animal_holdout/pandas/panda_00755.jpg'
    ]

    model = joblib.load('animals_image_classify_scikit_model.sav')

    for test_image in test_images:
        print("-----------------------------")
        predict_scene(model, test_image)
