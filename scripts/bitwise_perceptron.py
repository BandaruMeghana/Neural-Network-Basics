from modules.perceptron import Perceptron
import numpy as np
from config import alpha, epochs

# Create OR dataset
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
# Modify y to fit the XOR and notice that we can never classify accurately with different alpha, epochs.
# This is where we need deeper layers, for non-linear data
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

print("[INFO] Training the perceptron")
perceptron = Perceptron(X.shape[1], alpha)
perceptron.fit(X,y,epochs)

print("[INFO] Testing the perceptron")
for (x, target) in zip(X,y):
    pred = perceptron.predict(x)
    print("[INFO] data={}, ground-truth={}, pred={}".format(x, target[0],pred))