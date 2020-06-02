from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from networks.CONV.miniVGG import MiniVGG
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
from config import miniVGG_CIFAR
import matplotlib
from tensorflow.keras.callbacks import ModelCheckpoint
import os
matplotlib.use("Agg")

print("[INFO] Loading the CIFAR-10 dataset")
((trainX, trainY), (testX,testY)) = cifar10.load_data()

trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print("[INFO] Compiling the model...")
model = MiniVGG.build(width=32, height=32, depth=3, classes=10)
opt = SGD(miniVGG_CIFAR["alpha"], decay=miniVGG_CIFAR["alpha"]/miniVGG_CIFAR["epochs"], momentum=0.9, nesterov=True)
model.compile(loss=miniVGG_CIFAR["loss"], optimizer=opt, metrics=["accuracy"])

# Checkpointing the network & save only the *best* model to the disk
fname = os.path.sep.join([miniVGG_CIFAR["weights_path"], "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor='val_loss', mode=min, save_best_only=True,verbose=1)
callbacks = [checkpoint]


print("[INFO] Training the network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=miniVGG_CIFAR["batch_size"], epochs=miniVGG_CIFAR["epochs"],callbacks=callbacks, verbose=2)

print("[INFO] Evaluating the model...")
preds = model.predict(testX, batch_size=miniVGG_CIFAR["batch_size"])
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=labelNames))

# dispplaying the accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="training loss")
plt.plot(np.arange(0,40), H.history["val_loss"], label="validation loss")
plt.plot(np.arange(0,40), H.history["accuracy"], label="training accuracy")
plt.plot(np.arange(0,40), H.history["val_accuracy"], label="validation accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
