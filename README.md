# Machine Learning in Python

This repo is from the blog:

[Machine Learning in Python](https://www.pyimagesearch.com/2019/01/14/machine-learning-in-python/)


## Setup

- Must use python 3.6.x.  Python 3.7 will not work yet for Tensorflow

```python3.6 -m venv venv```

```
pip install numpy
pip install pillow
pip install --upgrade scikit-learn
pip install tensorflow
pip install keras
pip install --upgrade imutils
```

- imutils assumes opencv is installed.  If you are creating a 'clean' python 3.6.x virtual environment you wont have opencv installed.  To get around this, pull out the `paths.py` file and put this next to the scripts.



- imutils assumes opencv is installed.  If you are creating a 'clean' python 3.6.x virtual environment you wont have opencv installed.  To get around this, pull out the `paths.py` file and put this next to the scripts.

### classify_images

- Options

--dataset animals --model random_forest

360 Coast Pictures
328 forest Pictures
260 Highway Pictures

1001 Cats
1001 Dogs
1001 Pandas

### Create saved model
python create_image_classification_model.py --dataset animals --model all

python create_image_classification_model.py --dataset 3scenes --model all


## DeepLearning Keras CNN

In pyimagesearch/nn/conv run shallownet_animals.py.

This is an implementation of a very simple CNN learning model.

Interestingly, when run on the animals dataset the CNN model does worse than randomforest using just color attributes.

```text
[INFO] evaluating network...
              precision    recall  f1-score   support
         cat       0.63      0.26      0.37       236
         dog       0.52      0.79      0.62       263
       panda       0.85      0.85      0.85       250
   micro avg       0.64      0.64      0.64       749
   macro avg       0.67      0.63      0.61       749
weighted avg       0.66      0.64      0.62       749

```

With a Feedforward NN (by running  pyimagesearch/nn/keras_animals.py) the model results were:
```text
[INFO] evaluating network...
              precision    recall  f1-score   support
        cats       0.57      0.31      0.40       236
        dogs       0.52      0.71      0.60       263
      pandas       0.80      0.84      0.82       250
   micro avg       0.62      0.62      0.62       749
   macro avg       0.63      0.62      0.60       749
weighted avg       0.63      0.62      0.61       749

```

Where the RandomForest performed as:

```text
animals_image_classify_scikit_model.sav
('random_forest', 0.6916917922948074, RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=4, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False))
```