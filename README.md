## KU Machine Learning Neural Network Project
By **Jakob Wulf-Eck** and **Austin Irvine**

## Project Motivation
Build a neural network for two different datasets. We chose two different datasets with slightly different end-goals.

### MNIST Database
**Database of handwritten digits**
Our motivation was to be able to predict a **handwritten number/digit** with high accuracy from a 28x28 grayscale image.

### [HASYv2](https://mafiadoc.com/the-hasyv2-dataset-arxiv_5a0c183b1723dda02c95fe11.html) 
**Database of handwritten mathematical symbols**
Our motivation was to be able to predict a **handwritten mathematical symbol or letter** with high accuracy if given a 32x32 black-and-white image.

## Learning & Understanding
We used the first chapter of the book Neural Networks and Deep Learning from this website pdf source [BOOK](http://neuralnetworksanddeeplearning.com/chap1.html). In addition to this learning resource, we used knowledge from our class notes and other online blogs to build our neural network and calculus
based back propagation algorithm.

## Understanding the Datasets

### Understanding The Data
Both of the datasets we worked with contained images of handwriting, but in different contexts. The MNIST dataset purely worked with grayscale images of defined digits. On the other hand, the HASYv2 dataset worked with far more advanced mathematical and assorted symbols in black and white. The two datasets offered some interesting differences such as the vast number of pngs for the HASYv2 set verus the given pixel arrays from the MNIST dataset. We were able to use both datasets in similar ways with our neural net algorithm. An additional discovery was made when reviewing the paper produced from the HASYv2 dataset that the authors worked with both MNIST and HASYv2. The HASYv2 is far larger with 369 classes and over 168k instances to learn and train over. 
### HASYv2 and Git LFS
As the HASYv2 dataset is formatted as a collection of very large CSV files and a large folder of PNG images, we ran a script to simplify it to two (training and test) CSV files, which consist of only the images, expressed as an array of pixels, and the labels.
These CSV files are too large to fit on a Git repo by default. We have used Git LFS (Large File Storage) to commit them. To download them, ensure Git LFS is installed on your system and run `git lfs install` in your local repository before fetching/cloning. If the hasyTestSet and hasyTrainingSet files you obtain are pointer files instead of large, csv-formatted data files, try running `git lfs fetch`.

 Additionally, the full HASYv2 dataset is committed as a zip file.


## How To Run

**Run this for the MNIST Dataset**
```
python digitrecog.py
```

**Run this for the HASYv2 Dataset**
```
python hasvRecog.py
```

## Tech

<b>Built with</b>
- [Python](https://www.python.org/)

## Results
