from sklearn.externals import joblib
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


def predict_animal(imagePath):
    print(f'Predict for image: {imagePath}')
    model = joblib.load('animals_image_classify_scikit_model.sav')
    features = rgbHisto.get_features(imagePath)
    prediction = model.predict([features])

    print(f'Prediction: {prediction}')
    print(f'Targets: {load_targets()}')



if __name__ == '__main__':
    test_images = [
        './test_images/cat_9756.jpg',
        './test_images/cat_9829.jpg',
        './test_images/cat_9999.jpg',
        './test_images/dog_9744.jpg',
        './test_images/dog_9751.jpg',
        './test_images/dog_9969.jpg',
        './animal_holdout/cats/cats_00843.jpg',
        './animal_holdout/cats/cats_00997.jpg',
        './animal_holdout/dogs/dogs_00102.jpg',
        './animal_holdout/dogs/dogs_00163.jpg',
        './animal_holdout/pandas/panda_00050.jpg',
        './animal_holdout/pandas/panda_00755.jpg'
    ]

    for test_image in test_images:
        print("-----------------------------")
        predict_animal(test_image)
