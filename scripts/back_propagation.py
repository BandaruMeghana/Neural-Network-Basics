from modules.neural_network import NeuralNetwork
import numpy as np

# Create OR dataset
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

nn = NeuralNetwork([2,2,1], alpha=0.5)
print("[INFO] Training the network...")
nn.fit(X,y,epochs=20000)
print("[INFO] Testing the network...")
for (x,target) in zip(X,y):
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print("[INFO] data={}, ground-truth={}, pred={}, step={}".format(x, target[0], pred, step))