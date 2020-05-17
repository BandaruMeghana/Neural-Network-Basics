from modules.neural_network import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

# Load the MNIST dataset and apply min/max scaling to scale the pixel intensity values to the range [0, 1] (each
# image is represented as an 8x8 = 64-dim feature vector
digits = datasets.load_digits()
data = digits.data.astype('float')
data = (data - data.min()) / (data.max() - data.min())
print('[INFO]: Samples={}, Dimension={}'.format(data.shape[0], data.shape[1]))

# Construct the training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)

# Convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# Train the network
print('[INFO]: Training....')
nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])
print('[INFO]: {}'.format(nn))
nn.fit(trainX, trainY, epochs=1000)

# Test the network
print('[INFO]: Testing....')
predictions = nn.predict(testX)
predictions = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))
