import numpy as np

class Perceptron:
    def __init__(self, N, alpha):
        """
        :param N: size of input layer (or) number of features in each data point
        :param alpha: Learning rate
        :purpose: Initialize the random weights with Gaussian distribution(mean=0 and std.dev=1) and learning rate
        """
        self.W = np.random.rand(N+1)/np.sqrt(N) # N+1 to include the bias. sqrt(N) to scale the weights for faster convergence
        self.alpha = alpha

    def step_activation_function(self,x):
        return 1 if x>0 else 0

    def fit(self,X,y,epochs=10):
        """
        :param X: feature vector
        :param y: labels
        :param epochs: number of epochs for training
        :return:
        :purpose: Train the network
        """
        # add bias column to the X
        X = np.c_[X, np.ones([X.shape[0]])]
        for epoch in np.arange(0,epochs):
            for (x,target) in zip(X,y):
                p = self.step_activation_function(np.dot(x, self.W))
                # update the weight ONLY in case of misclassification
                if p!= target:
                    error = p - target
                    self.W += -self.alpha * error * x


    def predict(self, X, add_bias=True):
        """
        :param X: feature vector for the test data points
        :param add_bias: flag to indicate weather ot not add the bias column.
        :return: labels/predictions
        """
        # Ensure the X is a matrix
        X = np.atleast_2d(X)

        if add_bias:
            X = np.c_[X, np.ones((X.shape[0]))]
        return self.step_activation_function(np.dot(X,self.W))

