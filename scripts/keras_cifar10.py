from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from config import keras_CIFAR
import matplotlib.pyplot as plt
import numpy as np
from time import time

start = time()
print("[INFO] Loading the CIFAR10 dataset")
((trainX,trainY),(testX,testY)) = cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0
# flatten the input
trainX = trainX.reshape((trainX.shape[0],3072))
testX = testX.reshape((testX.shape[0], 3072))

# convert the labels from ints to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# initialize the label names for the CIFAR-10 dataset
label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Define the network architecture --> 3072-1024-512-10

model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

print("[INFO] Training the network...")
sgd = SGD(keras_CIFAR['alpha'])
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data=(testX,testY), epochs=keras_CIFAR['epochs'], batch_size=keras_CIFAR['batch_size'])


print("[INFO] Evaluating the network...")
preds = model.predict(testX, batch_size=keras_CIFAR['batch_size'])
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=label_names))


# plot
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
plt.savefig(keras_CIFAR['output_path'])
plt.show()

end = time()
print("[INFO] Time taken on CPU is {:.2f} min".format((end-start)/60))