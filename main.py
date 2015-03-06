import raop.pipeline as pipeline
import raop.model.ml as model
import numpy as np
import os

trainFile = 'resources/2-train-preprocessed-keys-added.json'

modelOutpath = 'resources/models/'

#fetch features and requestor results (i.e. X's and Y's)
features, pizzas = pipeline.getFeatures(trainFile)


#my model details
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
modelName = "GaussianNaiveBayes"
directoryName = "GaussianNaiveBayes"
description = "Steven Zimmerman - March 3rd 2015 - Gaussian Naive Bayes"
pipeline.modelPipeline(classifier, features, pizzas, modelOutpath, modelName, directoryName, description)

#svm
import sklearn.svm as svm 
classifier = svm.SVC()
modelName = "SVM"
directoryName = "SVM"
description = ""
pipeline.modelPipeline(classifier, features, pizzas, modelOutpath, modelName, directoryName, description)

#Decision Tree
from sklearn import tree
classifier = tree.DecisionTreeClassifier()
modelName = "Decision Tree"
directoryName = "DecisionTree"
description = ""

pipeline.modelPipeline(classifier, features, pizzas, modelOutpath, modelName, directoryName, description)

#Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
classifier = linear_model.LogisticRegression()
modelName = "Logistic Regression"
directoryName = "LogisticRegression"
description = ""

pipeline.modelPipeline(classifier, features, pizzas, modelOutpath, modelName, directoryName, description)

