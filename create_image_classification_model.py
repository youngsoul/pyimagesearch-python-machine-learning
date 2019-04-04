# USAGE
# python create_image_classification_model.py --dataset animals --model all
# default dataset=3scenes
# default model=knn
# python create_image_classification_model.py

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from path_utils import list_images
import argparse
import os
from rgbhistogram import RGBHistogram
from sklearn.externals import joblib
from xgboost import XGBClassifier
import pandas as pd


rgbHisto = RGBHistogram([8, 8, 8])

# define the dictionary of models our script can use, where the key
# to the dictionary is the name of the model (supplied via command
# line argument) and the value is the model itself
models = {
    "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
    "svm": SVC(kernel="linear"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_leaf=4),
    "mlp2": MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, alpha=0.0001,
                          solver='adam', verbose=10, tol=0.000000001),
    "mlp": MLPClassifier(),
    "xgboost": XGBClassifier(learning_rate=0.01)

}


"""
/Users/patrickryan/Development/python/mygithub/pyimagesearch-python-machine-learning/3scenes
--dataset 3scenes
--dataset animals

"""
def get_arguments():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", type=str, default="animals",
                    help="path to directory containing the '3scenes' dataset")
    ap.add_argument("-m", "--model", type=str, default="all",
                    help="type of python machine learning model to use")
    args = vars(ap.parse_args())

    return args


def get_image_features_and_labels(dataset_directory_name):
    # grab all image paths in the input dataset directory, initialize our
    # list of extracted features and corresponding labels
    print(f"[INFO] extracting image features from dataset: [{dataset_directory_name}]...")
    imagePaths = list_images(dataset_directory_name)
    feature_data = []
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
        feature_data.append(cv2_features)

        # extract the class label from the file path and update the
        # labels list
        # label is the directory name where the images reside.  the name of the image file does not matter
        label = imagePath.split(os.path.sep)[-2]
        image_labels.append(label)

    return feature_data, image_labels

# when you get here
# every row in the data array is
def one_hot_encode_targets(target_values, dataset_name):
    # encode the labels, converting them from strings to integers
    le = LabelEncoder()
    numeric_labels = le.fit_transform(target_values)
    print(le.classes_)
    with open(f'./{dataset_name}_labels.txt', 'w') as f:
        for i,target_name in enumerate(le.classes_):
            f.write(f"{i},{target_name}")
            f.write("\n")

    return numeric_labels, le.classes_


def cross_validate_model(model_name, X, y):
    # train the model
    # print("[INFO] using '{}' model".format(args["model"]))
    model = models[model_name]
    scores = cross_val_score(model, X, y, cv=5)
    accuracy = scores.mean()
    return model_name, accuracy, model


def run_model_by_name(model_name, trainX, trainY, testX, testY, y_classes):
    # train the model
    # print("[INFO] using '{}' model".format(args["model"]))
    model = models[model_name]
    model.fit(trainX, trainY)
    # print(f'Model: \n {model}')
    # make predictions on our data and show a classification report
    # print("[INFO] evaluating...")
    predictions = model.predict(testX)
    accuracy = accuracy_score(testY, predictions)
    class_report = classification_report(testY, predictions,
                                         target_names=y_classes)
    # print(f'Model: {model_name}')
    # print(f'Accuracy: {accuracy}')
    # print(f'Classification Report:\n{class_report}')
    return model_name, accuracy, model


def get_model_result_sort_key(x):
    return x[1]

if __name__ == '__main__':
    args = get_arguments()
    dataset_dir_name = args["dataset"]
    X,y = get_image_features_and_labels(dataset_dir_name)
    y_transformed, y_classes = one_hot_encode_targets(y, dataset_dir_name)

    X = pd.DataFrame(X)
    y_transformed = pd.DataFrame(y_transformed)
    # perform a training and testing split, using 75% of the data for
    # training and 25% for evaluation
    (trainX, testX, trainY, testY) = train_test_split(X, y_transformed,
                                                      test_size=0.25)

    model_name = args["model"]

    results = []
    if model_name == 'all':
        for k, v in models.items():
            # results.append(run_model_by_name(k, trainX, trainY, testX, testY, y_classes))
            results.append(cross_validate_model(k, X, y_transformed))

        sorted_models = sorted(results, key=get_model_result_sort_key, reverse=True)
        for model_result in sorted_models:
            print("----------------------------------")
            print(model_result)
            print("----------------------------------")

        print("Best Model")
        print(sorted_models[0])
        best_model = sorted_models[0][2]
        best_model.fit(X,y)
        saved_model_name = f"{dataset_dir_name}_image_classify_scikit_model.sav"
        with open(f'./{dataset_dir_name}_best_model_details.txt', 'w') as f:
            f.write(saved_model_name)
            f.write("\n")
            f.write(f"{sorted_models[0]}")

        print(f"Saving model to: {saved_model_name}")
        joblib.dump(best_model, saved_model_name)


    else:
        model_name, accuracy, model = cross_validate_model(model_name, X, y_transformed) #run_model_by_name(model_name, trainX, trainY, testX, testY, y_classes)
        print(model_name, accuracy)
        model.fit(X,y)
        saved_model_name = f"{dataset_dir_name}_image_classify_scikit_model.sav"
        joblib.dump(model, saved_model_name)
        print(f"Saving model to: {saved_model_name}")
        with open(f'./{dataset_dir_name}_{model_name}_model_details.txt', 'w') as f:
            f.write(model)

