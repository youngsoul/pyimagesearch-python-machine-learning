from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
from sklearn.model_selection import train_test_split


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False, default="../../animals",
                help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# resize image to 32x32
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor(flatten_array=True)

# load the data from disk then scale the raw pixel intensities to the range [0,1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)
labelNames = ['cats', 'dogs', 'pandas']

# define a deep network architecture
# 3072-1024-512-10
model = Sequential()
model.add(Dense(1024, input_shape=(32*32*3,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(3,activation='softmax'))

# train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# plot the training loss and accuracy
# plot the testing classification data
plt.style.use('ggplot')
plt.figure()
plt.title("Training Loss and Accuracy")
plt.plot(np.arange(0,100), H.history['loss'], label="train_loss")
plt.plot(np.arange(0,100), H.history['val_loss'], label="val_loss")
plt.plot(np.arange(0,100), H.history['acc'], label="train_acc")
plt.plot(np.arange(0,100), H.history['val_acc'], label="val_acc")

# construct a figure that plots the loss over time
plt.title('Training Loss')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()

plt.show()
