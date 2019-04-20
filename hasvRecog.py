import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random

def imgToArray(filename):
   img = Image.open(filename).convert('L')  # convert image to 8-bit grayscale
   WIDTH, HEIGHT = img.size 
   data = list(img.getdata()) # convert image data to a list of integers
   return data

def main():
    hasV = './HASYv2'

    # Commented Iteration Content For Classication Tasks
    # hasVCla = 'classification-task'
    # hasVVer = 'verification-task'
    
    # for i in range(1, 10):
        # trainingData = hasV + '/' + hasVCla + '/' + 'fold-' + str(i) + '/' + 'train.csv'
        # testData = hasV + '/' + hasVCla + '/' + 'fold-' + str(i) + '/' + 'test.csv'
        # trainingDF = pd.read_csv(trainingData)
        # testDF = pd.read_csv(testData)

    hasYData = hasV + '/' + 'hasy-data-labels.csv'
    hasyDF = pd.read_csv(hasYData)

    symbData = hasV + '/' + 'symbols.csv'
    symbDF = pd.read_csv(symbData)

    for item in hasyDF.index:
        img_path = hasyDF.ix[item][0]
        symbol = hasyDF.ix[item][1]
        latex = hasyDF.ix[item][2]
        u_id = hasyDF.ix[item][3]
        data = imgToArray(hasV+'/'+img_path)
        img_symb = (data, symbol)

        # img = mpimg.imread(hasV+'/'+img_path)
        # imgplot = plt.imshow(img)
        # plt.show()

main()