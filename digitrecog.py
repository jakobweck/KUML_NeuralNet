import numpy as np
import random
import pickle
import gzip
import os
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
def sigmoidPrime(x):
    return sigmoid(x)*(1-sigmoid(x))

class NeuralNet(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
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
            a = sigmoid(np.dot(w, a)+b)
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
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.delta(activations[-1], tOutput)
      #  delta = cost_derivative(activations[-1], tOutput) * \
       #     sigmoidPrime(zs[-1])
        biasGradient[-1] = delta
        weightGradient[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoidPrime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            biasGradient[-l] = delta
            weightGradient[-l] = np.dot(delta, activations[-l-1].transpose())
        return (biasGradient, weightGradient)
    def evaluate(self, test_data):
        test_results = [((self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
def cost_derivative(x, y):
    return x-y
def loadMNIST( prefix, folder ):
    intType = np.dtype( 'int32' ).newbyteorder( '>' )
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile( folder + "/" + prefix + '-images.idx3-ubyte', dtype = 'ubyte' )
    magicBytes, nImages, width, height = np.frombuffer( data[:nMetaDataBytes].tobytes(), intType )
    data = data[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, width, height ] )

    labels = np.fromfile( folder + "/" + prefix + '-labels.idx1-ubyte',
                          dtype = 'ubyte' )[2 * intType.itemsize:]

    return data, labels

def digitHotVector(x):
    res = np.zeros((10,1))
    res[x] = 1
    return res
def vectorMaxIndex(x):
    return x.argmax(0)

def pPrintArr(arr):
    for i in range(len(arr)):
        line = ""
        for j in range(len(arr[0])):
            if arr[i][j] != 0.0:
                line += "x"
            else:
                line += '.'
        print(line)

def pPrintInput(arr, ind):
    for i in range(len(arr[0][ind])):
        line = ""
        for j in range(len(arr[0][ind][0])):
            if arr[0][ind][i][j] != 0.0:
                line += "x"
            else:
                line += '.'
        print(line)

def main():
    trainingData = [[],[]]
    testData = [[],[]]
    trainingData[0], trainingData[1] = loadMNIST("train", "mnistData")
    testData[0],testData[1] = loadMNIST("t10k", "mnistData")


    trInputs = [np.reshape(x, (784,1)) for x in trainingData[0]]
    trInputs = [np.divide(x, 255.0) for x in trInputs] #NORMALIZE INPUT TO 0-1 RANGE VERY IMPORTANT
    trOutputs = [digitHotVector(y) for y in trainingData[1]]
    trainingData = list(zip(trInputs, trOutputs))

    testInputs = [np.reshape(x, (784,1)) for x in testData[0]]
    testData = list(zip(testInputs, testData[1]))


    net = NeuralNet([784, 30, 10]) #784 inputs (28x28px image), 30 hidden neurons, 10 outputs (0-9)
    net.sgd(trainingData, 30, 10, .5, testData=testData) #30 iterations, sgd batch size of 10, learning rate of .5

    done = False
    while not done:
        input("Press enter to test on a random member of the training set.")
        index = np.random.randint(0, len(trainingData))
        inputImg = trainingData[index][0]
        pPrintArr(np.reshape(inputImg, (28, 28)))
        desiredOutput = vectorMaxIndex(trainingData[index][1])
        res = net.feedforward(inputImg)
        print("Correct output: " + str(desiredOutput) +". NN output: " + str(res))

main()