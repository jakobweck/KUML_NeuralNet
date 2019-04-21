import numpy as np
import random
class NeuralNet(object):
    def __init__(self, sizes, useRelu):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.useRelu = useRelu
    def activation(self, x):
        if self.useRelu:
            return self.relu(x)
        else:
            return self.sigmoid(x)
    def activationPrime(self, x):
        if self.useRelu:
            return self.reluPrime(x)
        else:
            return self.sigmoidPrime(x)
    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))
    def sigmoidPrime(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def relu(self, x):
        return np.where(x > 0, x, 0.0)
    def reluPrime(self, x):
        return np.where(x > 0, 1.0, 0.0)

    #stochastic gradient descent 
    #train weights and biases by calculating change rate using random subsets of the training data
    def sgd(self, trainingData, numEpochs, miniBatchSize, learningRate, testData = None):
        n = len(trainingData)
        if testData: n_test = len(testData)
        for i in range(numEpochs):
            random.shuffle(trainingData)
            miniBatches = [trainingData[j:j+miniBatchSize] for j in range(0, n, miniBatchSize)]
            for batch in miniBatches:
                self.trainMiniBatch(batch, learningRate)
            if testData:
                print ("Epoch: " + str(i) + " " + str(self.evaluate(testData)) +'/' + str(n_test))
            print("Epoch " + str(i) + " complete")
    def trainMiniBatch(self, miniBatch, learningRate):
        biasGradient = [np.zeros(b.shape) for b in self.biases]
        weightGradient = [np.zeros(w.shape) for w in self.weights]
        #generate gradient delta that minimizes cost for mini batch using backpropagation
        #update gradients
        for x, y in miniBatch:
            biasGradientDelta, weightGradientDelta = self.backprop(x, y)
            biasGradient = [bg+delta for bg, delta in zip(biasGradient, biasGradientDelta)]
            weightGradient = [wg+delta for wg, delta in zip(weightGradient, weightGradientDelta)]
        #step weights down the gradient using a step size of learningRate
        self.weights = [w-(learningRate/len(miniBatch))*wg 
                        for w, wg in zip(self.weights, weightGradient)]
        self.biases = [b-(learningRate/len(miniBatch))*bg 
                       for b, bg in zip(self.biases, biasGradient)]
    #calculate network output for given input
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.activation(np.dot(w, a)+b)
        return a.argmax(0)
    def delta(self, a, b):
        return (a-b)
    def backprop(self, tInput, tOutput):
        biasGradient = [np.zeros(b.shape) for b in self.biases]
        weightGradient = [np.zeros(w.shape) for w in self.weights]
        activation = tInput
        activations = [tInput] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.activation(z)
            activations.append(activation)
        # backward pass
        delta = self.delta(activations[-1], tOutput)
      #  delta = cost_derivative(activations[-1], tOutput) * \
       #     sigmoidPrime(zs[-1])
        biasGradient[-1] = delta
        weightGradient[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activationPrime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            biasGradient[-l] = delta
            weightGradient[-l] = np.dot(delta, activations[-l-1].transpose())
        return (biasGradient, weightGradient)
    def evaluate(self, test_data):
        test_results = [((self.feedforward(x)), y) for (x, y) in test_data]
        # for (x,y) in test_results:
        #     if (x==y): print(x)
        return sum(int(x == y) for (x, y) in test_results)
