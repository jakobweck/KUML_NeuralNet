import numpy as np
import random
import pickle
import gzip
import os
import network
    
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


    net = network.NeuralNet([784, 30, 10]) #784 inputs (28x28px image), 30 hidden neurons, 10 outputs (0-9)
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