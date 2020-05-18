# Neural-Network-Basics

This repository covers the implementation of 2 basic concepts of Neural Networks. 
1. Perceptron
2. FeedForward and back propagation

The perceptron concept is illustrated by the XOR dataset i.e., the data is constructed on the basis of XOR bitwise operators. 

Upon running the ```bitwise_perceptron.py``` multiple times, we see the correct predictions, for OR, AND datasets. But for XOR, we never get the correct predections in case of perceptron. This is because, OR, AND are linear datasets but XOR is a non-linear dataset. This example helps in illustrating the need for mutliple hidden layers, especially to handle the non-linear datasets which are the majority of the datasets in the current world. 

![Data Linearity](https://github.com/BandaruMeghana/Neural-Network-Basics/blob/master/data_linearity.PNG)

All the core functionality of perceptron and back propagation are implemented in ```perceptron.py``` and ```neural_network.py``` files -inside the ```modules``` directory. The functionalities are implemented as classes.

Use the files inside ```scripts``` to see the functionalities in action. 

Although, implementing the core concepts from scratch gives us the intuition, it is operationally easy to use the available libraries. 
Use the ```keras_MNIST.py``` and ```keras_CIFAR10.py``` to see the Tensorflow's Keras implementation. 
- The keras implementation of MNIST dataset yields us an accuracy of ~92%. This is not the best we can achieve. Using the Convolution Neural Netwroks (CNN)s give us ~99% accuracy. 

![Loss on CIFAR10 dataset](https://github.com/BandaruMeghana/Neural-Network-Basics/blob/master/CIFAR.png)
- The CIFAR dataset is color based. Here, we can see that, with pure neural nets, we get an accuracy of ~56%. 
- We can see that after epoch 10, the model is overfitting. i.e., although the training loss reduced over the epochs, the validation loss keeps on increasing. 
__Basic feedforward networks with strictly fully-connected layers are not suitable for challenging image datasets.__

The time taken tp process the MNIST and CIFAR on CPU is 9.80 min and 56.49 min respectively. Upon using the GPU's from Google Colab, the time is reduced to 3.77min and 14.71min!!!

Finally, ```config.py``` has all the parameters that can be played around with!
