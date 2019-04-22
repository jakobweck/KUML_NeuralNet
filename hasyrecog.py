import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random
import hasyNetwork
import re
import sys

def imgToArray(filename):
   img = Image.open(filename).convert('L')  # convert image to 8-bit grayscale
   data = list(img.getdata()) # convert image data to a list of integers
   return data
def pPrintArr(arr):
    for i in range(len(arr)):
        line = ""
        for j in range(len(arr[0])):
            if arr[i][j] != 0.0:
                line += "x"
            else:
                line += '.'
        print(line)
def binNot(x):
    if x==1: return 0
    if x==0: return 1
    return None

def digitHotVector(x):
    res = np.zeros((369,1))
    res[x] = 1
    return res
def vectorMaxIndex(x):
    return x.argmax(0)
def main():
    if(len(sys.argv)<6):
        numHiddenLayers = 1
        layers = [1024, 30, 30, 369]
        epochs = 200
        learningRate = .8
        lmbda = 0 
        batchSize = 1
    else:
        numHiddenLayers = int(sys.argv[1])
        layers = [1024]
        for i in range(2, numHiddenLayers+2):
            layers.append(int(sys.argv[i]))
        layers.append(369)
        currArg = 2+numHiddenLayers
        epochs = int(sys.argv[currArg])
        currArg +=1
        learningRate = float(sys.argv[currArg])
        currArg +=1
        lmbda = float(sys.argv[currArg])
        currArg += 1
        batchSize = int(sys.argv[currArg])
    
    symbolsDf = pd.read_csv("hasy/symbols.csv")
    symbols = dict()
    symbolNames = dict()
    for item in symbolsDf.index:
        name = symbolsDf.ix[item][1]
        theirId = symbolsDf.ix[item][0]
        myId = symbolsDf.ix[item][4]
        symbols[theirId] = myId
        symbolNames[myId] = name
    trainingSet = np.loadtxt("hasyTestSet.csv", delimiter=",")
    finalTrainingSet = []
    for arr in trainingSet:
        label, arr = arr[-1], arr[:-1]
        arr = np.divide(arr, 255.0)
        arr = np.array([binNot(x) for x in arr])
        arr = np.reshape(arr, (1024,1))
        label = digitHotVector(symbols[label])
        finalTrainingSet.append((arr, (label)))
    testSet = np.loadtxt("hasyTestSet.csv", delimiter=",")
    finalTestSet = []
    for arr in testSet:
        label, arr = arr[-1], arr[:-1]
        arr = np.divide(arr, 255.0)
        arr = np.array([binNot(x) for x in arr])
        arr = np.reshape(arr, (1024,1))
        label = (symbols[label])
        finalTestSet.append((arr, (label)))
    net = hasyNetwork.NeuralNet(layers)
    net.sgd(finalTrainingSet, epochs, batchSize, learningRate, lmbda=lmbda)
    done = False
    while not done:
        input("Press enter to test on a random member of the training set.")
        index = np.random.randint(0, len(finalTrainingSet))
        inputImg = finalTrainingSet[index][0]
        pPrintArr(np.reshape(inputImg, (32, 32)))
        desiredOutput = symbolNames[vectorMaxIndex(finalTrainingSet[index][1])[0]]
        res = symbolNames[net.feedforward(inputImg).argmax(0)[0]]
        print("Correct output: " + str(desiredOutput) +". NN output: " + str(res))

main()