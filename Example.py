import raop.pipeline as pipeline
import raop.model.ml as model
import numpy as np
import os

trainFile = 'resources/2-train-preprocessed-keys-added.json'

modelOutpath = 'resources/models/'

#fetch features and requestor results (i.e. X's and Y's)
features, pizzas = pipeline.getFeatures(trainFile,0,[1,2,3,4])

#normalize
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(features)
features = scaler.transform(features)

#my model details
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#modelName = "GaussianNaiveBayes-Norm-AddRBJJ"
#directoryName = "GaussianNaiveBayes"
#description = "Steven Zimmerman - 8th March 2015  - Gaussian Naive Bayes"
#pipeline.modelPipeline(classifier, features, pizzas, modelOutpath, modelName, directoryName, description)

#svm
import sklearn.svm as svm 
classifier = svm.SVC(kernel='linear',class_weight = 'auto')
modelName = "SVM-Norm-linear-CLauto-Add-JJ-RB-NN-VB"
directoryName = "SVM"
description = "8th March 2015 "
pipeline.modelPipeline(classifier, features, pizzas, modelOutpath, modelName, directoryName, description)



#RFC
#from sklearn.ensemble import RandomForestClassifier
#classifier =RandomForestClassifier()
#modelName = "RandomForestClassifier-Norm-AddRBJJ"
#directoryName = "RandomForestClassifier"
#description = "Can Udomcharoenchaikit - 8th March 2015 - Random Forest"


#pipeline.modelPipeline(classifier, features, pizzas, modelOutpath, modelName, directoryName, description)

