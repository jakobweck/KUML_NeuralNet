"""network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np


#### Define the quadratic and cross-entropy cost functions

#### Main Network class
class NeuralNet(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def costFn(self, a, y):
        return 0.5*np.linalg.norm(a-y)**2

    def delta(self, z, a, y):
        return (a-y) * sigmoid_prime(z)

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None):

        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            print("Epoch %s training complete" % j)
            cost = self.total_cost(training_data, lmbda)
            training_cost.append(cost)
            print("Cost on training data: {}".format(cost))
            accuracy = self.accuracy(training_data, convert=True)
            training_accuracy.append(accuracy)
            print("Accuracy on training data: {} / {}".format(accuracy, n))

    def update_mini_batch(self, mini_batch, learningRate, lmbda, n):
        biasGradient = [np.zeros(b.shape) for b in self.biases]
        weightGradient = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            biasGradientDelta, weightGradientDelta = self.backprop(x, y)
            biasGradient = [nb+dnb for nb, dnb in zip(biasGradient, biasGradientDelta)]
            weightGradient = [nw+dnw for nw, dnw in zip(weightGradient, weightGradientDelta)]
        self.weights = [(1-learningRate*(lmbda/n))*w-(learningRate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, weightGradient)]
        self.biases = [b-(learningRate/len(mini_batch))*nb
                       for b, nb in zip(self.biases, biasGradient)]

    def backprop(self, x, y):
        biasGradient = [np.zeros(b.shape) for b in self.biases]
        weightGradient = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.delta(zs[-1], activations[-1], y)
        biasGradient[-1] = delta
        weightGradient[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            biasGradient[-l] = delta
            weightGradient[-l] = np.dot(delta, activations[-l-1].transpose())
        return (biasGradient, weightGradient)

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy

    def total_cost(self, data, lmbda):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            cost += self.costFn(a, y)/len(data)
            cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) # '**' - to the power of.
        return cost

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
