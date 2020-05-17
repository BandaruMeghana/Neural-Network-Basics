import numpy as np

class NeuralNetwork:

    def __init__(self, layers, alpha=0.1):
        """
        :param layers: list of integers which represents the actual architecture of the feedforward network. For example, a value of [2;2;1]
        :param alpha: learning rate for weight updates during the backward propagation
        :return: N/A
        :purpose: intialize the weights,
        """
        self.W = [] #list of weights for each layer
        self.layers = layers
        self.alpha = alpha

        # Initialize the weights that connect the current layer (i) to the next layer (i+1) and add the bias
        '''
        we are stopping iterating before the last 2 layers because, 
        for the (n-1)th layer, bias needs to be added only to the ith layer, not to the (i+1)th layer. 
        And, for the output layer, bias shouldn't be added.
        '''
        for i in np.arange(len(layers) - 2):
            w = np.random.rand(layers[i]+1, layers[i+1] + 1)
            self.W.append(w/np.sqrt(layers[i]))
        #adding bias to (n-1)th layer
        w = np.random.rand(layers[-2] + 1, layers[-1])
        self.W.append(w/np.sqrt(layers[-2]))


    def __repr__(self):
        """
        :purpose: for debugging
        :return: the neural network architecture
        """
        return "Neural Network {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))

    def sigmoid_derivative(self,x):
        return x*(1-x)

    def fit_partial(self, x, y):
        """
        :param x: individual data point
        :param y: data point's corresponding label
        :return:
        """
        # activation_outs of the last layer will be the predictions.
        activation_outs = [np.atleast_2d(x)] # stores the output activations for each layer. Intializing this with x

        # forward propagation
        for layer in np.arange(0, len(self.W)):
            # calculate the "net input" - the dot prod of x and W
            net = activation_outs[layer].dot(self.W[layer])
            # calculate the "net output" - applying the non-linear activation functions
            out = self.sigmoid(net)
            activation_outs.append(out)

        # Backward propagation
        # step-1: Calculate the difference between the prediction and true values
        error = activation_outs[-1] - y
        #step-2: Apply the chain rule and calculate the partial derivates(deltas) of the error w.r.t the weights
        deltas = [error * self.sigmoid_derivative(activation_outs[-1])]
        # step-3: update the weights(W) using deltas
        for layer in np.arange(len(activation_outs)-2, 0, -1):
            delta = deltas[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_derivative(activation_outs[layer])
            deltas.append(delta)

        # reverse the deltas
        deltas = deltas[::-1]

        # update the weights
        for layer in np.arange(0, len(self.W)):
            gradient = activation_outs[layer].T.dot(deltas[layer])
            self.W[layer] += -self.alpha * gradient


    def predict(self,X,addBias=True):
        # initialize the output predictions as the input features
        p = np.atleast_2d(X)
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
        return p


    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        # calculate the sum squared error
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        return loss

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        """
        :param X:
        :param y:
        :param epochs:
        :param displayUpdate:
        :purpose: for each epoch, loop over individual data point, make the prediction, compute the back propogation & update the weights
        :return:
        """
        # add bias column to X
        X = np.c_[X, np.ones((X.shape[0]))]
        for epoch in np.arange(0,epochs):
            for (x,target) in zip(X,y):
                self.fit_partial(x,target)

            if epoch == 0 or (epoch+1)%displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.2f}".format(epoch+1, loss))
