# USAGE
# python classify_images.py
# python classify_images.py --model svm

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image
from path_utils import list_images
import numpy as np
import argparse
import os
from rgbhistogram import RGBHistogram
from sklearn.externals import joblib
from more_itertools import unique_everseen
from operator import itemgetter
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import cross_val_score

def extract_color_stats(image):
    # split the input image into its respective RGB color channels
    # and then create a feature vector with 6 values: the mean and
    # standard deviation for each of the 3 channels, respectively
    (R, G, B) = image.split()
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
                np.std(G), np.std(B)]

    # return our set of features
    return features


"""
--dataset /Users/patrickryan/Development/python/mygithub/pyimagesearch-python-machine-learning/3scenes

"""
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="3scenes",
                help="path to directory containing the '3scenes' dataset")
ap.add_argument("-m", "--model", type=str, default="knn",
                help="type of python machine learning model to use")
args = vars(ap.parse_args())

rgbHisto = RGBHistogram([8, 8, 8])

# define the dictionary of models our script can use, where the key
# to the dictionary is the name of the model (supplied via command
# line argument) and the value is the model itself
models = {
    "knn": KNeighborsClassifier(n_neighbors=3),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
    "svm": SVC(kernel="linear"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=200, criterion='gini', min_samples_leaf=4),
    "mlp2": MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, alpha=0.0001,
                          solver='adam', verbose=10, tol=0.000000001),
    "mlp": MLPClassifier(),
    "xgboost": XGBClassifier(learning_rate=0.01)
}

dataset = args["dataset"]
# grab all image paths in the input dataset directory, initialize our
# list of extracted features and corresponding labels
print(f"[INFO] extracting image features from dataset: [{dataset}]...")
imagePaths = list_images(dataset)
data = []
image_labels = []

# loop over our input images
for imagePath in imagePaths:
    # load the input image from disk, compute color channel
    # statistics, and then update our data list
    # image = Image.open(imagePath)

    # using color stats does help along with rgbhisto
    # features = extract_color_stats(image)
    # data.append(features)

    # Depending upon the algorithm, using the histogram is helpful
    # check out mlp with and without histogram
    # check out random forest with and without
    cv2_features = rgbHisto.get_features(imagePath)
    data.append(cv2_features)

    # extract the class label from the file path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    image_labels.append(label)

# when you get here
# every row in the data array is

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(image_labels)

label_classes =  le.classes_
with open(f'./{dataset}_scene_labels.txt', 'w') as f:
    for i, y in enumerate(label_classes):
        f.write(f"{i},{y}")
        f.write("\n")


# unique_labels = list(unique_everseen(labels))
# unique_image_labels = list(unique_everseen(image_labels))
# print(unique_labels)
# print(unique_image_labels)
# with open(f'./{dataset}_scene_labels.txt', 'w') as f:
#     for x, y in list(zip(unique_labels, unique_image_labels)):
#         f.write(f"{x},{y}")
#         f.write("\n")

# perform a training and testing split, using 75% of the data for
# training and 25% for evaluation
data_df = pd.DataFrame(data)
labels_df = pd.DataFrame(labels)


# print("******")
# print(cross_val_score(XGBClassifier(learning_rate=0.01), data_df, labels_df, cv=3))

(trainX, testX, trainY, testY) = train_test_split(data_df, labels_df,
                                                  test_size=0.25)




model_name = args["model"]


def run_model_by_name(model_name):
    # train the model
    print("[INFO] using '{}' model".format(args["model"]))
    model = models[model_name]
    model.fit(trainX, trainY)
    print(f'Model: \n {model}')
    # make predictions on our data and show a classification report
    print("[INFO] evaluating...")
    predictions = model.predict(testX)
    accuracy = accuracy_score(testY, predictions)
    class_report = classification_report(testY, predictions,
                                         target_names=le.classes_)
    print(f'Model: {model_name}')
    print(f'Accuracy: {accuracy}')
    print(f'Classification Report:\n{class_report}')
    return model_name, accuracy, model


results = []
if model_name == 'all':
    for k, v in models.items():
        results.append(run_model_by_name(k))
        for r in results:
            print(r)

    results.sort(key=itemgetter(1))
    print(f"Best Model: {results[-1]}")

else:
    model_name, accuracy, model = run_model_by_name(model_name)
    print(model_name, accuracy)
    saved_model_name = f"{dataset}_image_classify_scikit_model.sav"
    joblib.dump(model, saved_model_name)
    print(f"Saving model to: {saved_model_name}")

