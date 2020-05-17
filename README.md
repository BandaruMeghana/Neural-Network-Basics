# Neural-Network-Basics

This repository covers the implementation of 2 basic concepts of Neural Networks. 
1. Perceptron
2. FeedForward and back propagation

The perceptron concept is illustrated by the XOR dataset i.e., the data is constructed on the basis of XOR bitwise operators. 

Upon running the ```bitwise_perceptron.py``` multiple times, we see the correct predictions, for OR, AND datasets. But for XOR, we never get the correct predections in case of perceptron. This is because, OR, AND are linear datasets but XOR is a non-linear dataset. This example helps in illustrating the need for mutliple hidden layers, especially to handle the non-linear datasets which are the majority of the datasets in the current world. 

<img src='data_linearity.png'>

All the core functionality of perceptron and back propagation are implemented in ```perceptron.py``` and ```neural_network.py``` files -inside the ```modules``` directory. The functionalities are implemented as classes.

Use the files inside ```scripts``` to see the functionalities in action. 

```config.py``` has all the parameters that can be played around with!
