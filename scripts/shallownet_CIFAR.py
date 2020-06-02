from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from networks.CONV.shallow_net import ShallowNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
from config import shallownet_CIFAR

print("[INFO] Loading the data")
((trainX,trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print("[INFO] Compiling the model")
opt = SGD(shallownet_CIFAR["alpha"])
model = ShallowNet.build(width=32, height=32, depth=32, classes=10)
model.compile(loss=shallownet_CIFAR["loss"], optimizer=opt, metrics=["accuracy"])

print("[INFO] Training the network...")
H = model.fit(trainX, trainY, validation_data=(testX,testY), batch_size=shallownet_CIFAR["batch_size"], epochs=shallownet_CIFAR["epochs"], verbose=1)

print("[INFO] Evaluating the network...")
preds = model.predict(testX, batch_size=shallownet_CIFAR["batch_size"])
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=labelNames))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,40), H.history["loss"], label="training loss")
plt.plot(np.arange(0,40), H.history["val_loss"], label="validation loss")
plt.plot(np.arange(0,40), H.history["accuracy"], label="training accuracy")
plt.plot(np.arange(0,40), H.history["val_accuracy"], label="validation accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()