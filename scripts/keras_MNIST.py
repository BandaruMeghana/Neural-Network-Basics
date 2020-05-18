from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential # FeedForward & add the layers sequentially
from tensorflow.keras.layers import Dense # Implements fully connected network
from tensorflow.keras.optimizers import SGD
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from config import keras_MNIST
import cv2
from time import time

start = time()
print("[INFO] Loading the MNIST dataset")
dataset = fetch_openml('mnist_784') # Downloads the 55MB Dataset as 784D flattened vector. Image size = 28 * 28
print("[INFO] Dataset shape: {}".format(dataset.data.shape))

# scale the raw pixels [0,255] to the range [0,1.0]
data = dataset.data.astype('float')/255.0

# train-test split
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size=0.25)

# convert the label ints to vectors
# ex: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] represents 3
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# define the network architecture --> 784-256-128-10
model = Sequential()
model.add(Dense(256, input_shape=(784, ), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

print("[INFO] Training the netwwork...")
sgd = SGD(keras_MNIST['alpha'])
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX,testY), epochs=keras_MNIST['epochs'], batch_size=keras_MNIST['batch_size'])
print("*****************", H.history)
print("[INFO] Evaluating the network...")
preds = model.predict(testX, batch_size=keras_MNIST['batch_size'])
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),target_names=[str(x) for x in lb.classes_]))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"], label="training loss")
plt.plot(np.arange(0,100), H.history["val_loss"], label="validation loss")
plt.plot(np.arange(0,100), H.history["accuracy"], label="training accuracy")
plt.plot(np.arange(0,100), H.history["val_accuracy"], label="validation accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(keras_MNIST['output_path'])
plt.show()

end = time()
print("[INFO] Time taken on CPU is {:.2f} min".format((end-start)/60))
