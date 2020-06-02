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

shallownet_CIFAR = {
    "dataset_path": "../data/animals/*/*.jpg",
    "alpha": 0.01,
    "loss": "categorical_crossentropy",
    "batch_size": 32,
    "epochs": 40
}

lenet_MNIST = {
    "alpha": 0.01,
    "loss": "categorical_crossentropy",
    "batch_size": 128,
    "epochs": 20
}

miniVGG_CIFAR = {
    "alpha": 0.01,
    "loss": "categorical_crossentropy",
    "batch_size": 64,
    "epochs": 40,
    "weights_path": "../weights"
}