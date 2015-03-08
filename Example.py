import raop.pipeline as pipeline
import raop.model.ml as model
import numpy as np
import os

trainFile = 'resources/2-train-preprocessed-keys-added.json'

modelOutpath = 'resources/models/'

#fetch features and requestor results (i.e. X's and Y's)
features, pizzas = pipeline.getFeatures(trainFile,0)

#normalize
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(features)
features = scaler.transform(features)
#min_max_scaler = preprocessing.MinMaxScaler()
#features = min_max_scaler.fit_transform(features)

#my model details
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
modelName = "GaussianNaiveBayes-Norm"
directoryName = "GaussianNaiveBayes"
description = "Steven Zimmerman - March 3rd 2015 - Gaussian Naive Bayes"
pipeline.modelPipeline(classifier, features, pizzas, modelOutpath, modelName, directoryName, description)

#svm
import sklearn.svm as svm 
classifier = svm.SVC(kernel='linear',class_weight = 'auto')
modelName = "SVM-Norm-linear-CLauto-timeHardCode"
directoryName = "SVM"
description = ""
pipeline.modelPipeline(classifier, features, pizzas, modelOutpath, modelName, directoryName, description)

#Decision Tree
from sklearn import tree
classifier = tree.DecisionTreeClassifier()
modelName = "Decision Tree-Norm"
directoryName = "DecisionTree"
description = ""

pipeline.modelPipeline(classifier, features, pizzas, modelOutpath, modelName, directoryName, description)

#Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
classifier = linear_model.LogisticRegression()
modelName = "Logistic Regression-Norm"
directoryName = "LogisticRegression"
description = ""

pipeline.modelPipeline(classifier, features, pizzas, modelOutpath, modelName, directoryName, description)
#RFC
from sklearn.ensemble import RandomForestClassifier
classifier =RandomForestClassifier()
modelName = "RandomForestClassifier-Norm"
directoryName = "RandomForestClassifier"
description = "Can Udomcharoenchaikit - 5th March 2015 - Random Forest"


pipeline.modelPipeline(classifier, features, pizzas, modelOutpath, modelName, directoryName, description)

