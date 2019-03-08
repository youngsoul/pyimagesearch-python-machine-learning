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
