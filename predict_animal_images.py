from sklearn.externals import joblib
from PIL import Image
import numpy as np
from rgbhistogram import RGBHistogram
from pathlib import Path

"""
Predict a large number of animal images by pointing to a directory or a single class and running a 
prediction for all files in directory
"""

rgbHisto = RGBHistogram([8, 8, 8])


def load_targets():
    targets = []
    with open('./animals_scene_labels.txt', 'r') as f:
        line = f.readline().strip()
        while line:
            targets.append(line)
            line = f.readline().strip()

    return targets


def predict_image(model, imagePath):
    print(f'Predict for image: {imagePath}')
    features = rgbHisto.get_features(imagePath)
    if features:
        prediction = model.predict([features])

        print(f'Prediction: {prediction}')
        print(f'Targets: {load_targets()}')
        return prediction[0]
    else:
        print(f"No features for image: {imagePath}")


if __name__ == '__main__':
    # cats, dogs, panda
    predicted_class = 'cats'
    cat_path = "/Volumes/MacBackup/CATS_DOGS_ORIGINAL/test/CAT"
    dog_path = "/Volumes/MacBackup/CATS_DOGS_ORIGINAL/test/DOG"
    image_path = cat_path


    test_images = [
        '/Volumes/MacBackup/CATS_DOGS_ORIGINAL/test/CAT/10125.jpg'
    ]

    p = Path(image_path)
    test_images = list(p.glob("*.jpg"))

    model = joblib.load('animals_image_classify_scikit_model.sav')

    true_count = 0
    for test_image in test_images:
        print("-----------------------------")
        prediction = predict_image(model, str(test_image))
        if prediction == predicted_class:
            true_count += 1

    print(f"Accuracy = {true_count/len(test_images)}")


