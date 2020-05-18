alpha = 0.01 #Learning rate
epochs = 20

# MNIST Keras implementation
keras_MNIST = {
    "output_path": '../outputs', # to save the loss and accuracy over time
    "alpha" : 0.01,
    "epochs": 100,
    "batch_size": 128
}

keras_CIFAR = {
    "output_path": '../CIFAR',
    "alpha": 0.01,
    "epochs": 100,
    "batch_size": 32
}