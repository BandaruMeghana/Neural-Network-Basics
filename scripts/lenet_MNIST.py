from networks.CONV.lenet import LeNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from sklearn import datasets
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from config import lenet_MNIST

print("[INFO] Loading MNIST dataset")
dataset = datasets.fetch_openml('mnist_784')
data = dataset.data

if K.image_data_format() == 'channel_first':
    data = data.reshape(data.shape[0],1,28,28)
else:
    data = data.reshape(data.shape[0], 28,28,1)

# train-test split
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target.astype("int"), test_size=0.25, random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

print("[INFO] Compiling the model...")
opt = SGD(lenet_MNIST["alpha"])
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss=lenet_MNIST["loss"], optimizer=opt, metrics=["accuracy"])

print("[INFO] Training the network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=lenet_MNIST["batch_size"], epochs=lenet_MNIST["epochs"], verbose=1)

print("[INFO] Evaluating the network...")
preds = model.predict(testX, batch_size=lenet_MNIST["batch_size"])
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))


plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="training loss")
plt.plot(np.arange(0,20), H.history["val_loss"], label="validation loss")
plt.plot(np.arange(0,20), H.history["accuracy"], label="training accuracy")
plt.plot(np.arange(0,20), H.history["val_accuracy"], label="validation accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()