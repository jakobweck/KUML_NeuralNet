import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random
import network
import re

def imgToArray(filename):
   img = Image.open(filename).convert('L')  # convert image to 8-bit grayscale
   WIDTH, HEIGHT = img.size 
   data = list(img.getdata()) # convert image data to a list of integers
   return data

def stripBracket(inputStr):
    return re.sub("\[]", '', inputStr.decode('utf-8'))

def main():
    # hasV = 'hasy'

    # Commented Iteration Content For Classication Tasks
    # hasVCla = 'classification-task'
    # hasVVer = 'verification-task'
    
    # for i in range(1, 10):
        # trainingData = hasV + '/' + hasVCla + '/' + 'fold-' + str(i) + '/' + 'train.csv'
        # testData = hasV + '/' + hasVCla + '/' + 'fold-' + str(i) + '/' + 'test.csv'
        # trainingDF = pd.read_csv(trainingData)
        # testDF = pd.read_csv(testData)

    # hasYData = hasV + '/' + 'hasy-data-labels.csv'
    # hasyDF = pd.read_csv(hasYData)
    
    # testData = hasV + '/' + hasVCla + '/' + 'fold-' + str(1) + '/' + 'test.csv'
    # testDF = pd.read_csv(testData)

    # trainingSet = []
    # testSet = []
    # symbolMapping = dict()

    # for item in hasyDF.index:
    #     img_path = hasyDF.ix[item][0]
    #     symbol = hasyDF.ix[item][1]
    #     latex = hasyDF.ix[item][2]
    #     if not symbol in symbolMapping:
    #         symbolMapping[symbol] = latex
    #     u_id = hasyDF.ix[item][3]
    #     data = imgToArray(hasV+'/'+img_path)
    #     img_symb = (data, symbol)
    #     trainingSet.append(img_symb)
    #     # img = mpimg.imread(hasV+'/'+img_path)
    #     # imgplot = plt.imshow(img)
    #     # plt.show()
    # for item in testDF.index:
    #     img_path = hasyDF.ix[item][0]
    #     symbol = hasyDF.ix[item][1]
    #     latex = hasyDF.ix[item][2]
    #     u_id = hasyDF.ix[item][3]
    #     data = imgToArray(hasV+'/'+img_path)
    #     img_symb = (data, symbol)
    #     testSet.append(img_symb)
    # np.savetxt("hasyTrainingSet.csv", trainingSet, delimiter=',', fmt="%s")
    # np.savetxt("hasyTestSet.csv", testSet, delimiter=',', fmt="%s")

    symbolsDf = pd.read_csv("hasy/symbols.csv")
    symbols = dict()
    for item in symbolsDf.index:
        name = symbolsDf.ix[item][1]
        theirId = symbolsDf.ix[item][0]
        myId = symbolsDf.ix[item][4]
        symbols[theirId] = myId
    trainingSet = np.loadtxt("hasyTestSet.csv", delimiter=",")
    finalTrainingSet = []
    for arr in trainingSet:
        label, arr = arr[-1], arr[:-1]
        arr = np.divide(arr, 255.0)
        arr = np.reshape(arr, (1024,1))
        label = symbols[label]
        finalTrainingSet.append((arr, int(label)))
    testSet = np.loadtxt("hasyTestSet.csv", delimiter=",")
    finalTestSet = []
    for arr in testSet:
        label, arr = arr[-1], arr[:-1]
        arr = np.divide(arr, 255.0)
        arr = np.reshape(arr, (1024,1))
        label = symbols[label]
        finalTestSet.append((arr, int(label)))
    net = network.NeuralNet([1024, 100, 369]) #1024 inputs (32x32 image), 30 hidden neurons, 369 outputs
    net.sgd(finalTrainingSet, 30, 10, .001, testData=finalTestSet)
main()