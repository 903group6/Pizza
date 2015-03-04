import raop.pipeline as pipeline
import raop.model.ml as model
import numpy as np
import os

trainFile = 'resources/2-train-preprocessed-keys-added.json'

modelOutpath = 'resources/models/'

#fetch features and requestor results (i.e. X's and Y's)
features, pizzas = pipeline.getFeatures(trainFile)
#]features = np.array(features) 
#pizzas = np.array(pizzas)

#my model details
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
modelName = "GaussianNaiveBayes"
directoryName = "GaussianNaiveBayes"
description = "Steven Zimmerman - March 3rd 2015 - Gaussian Naive Bayes"


pipeline.modelPipeline(classifier, features, pizzas, modelOutpath, modelName, directoryName, description)

