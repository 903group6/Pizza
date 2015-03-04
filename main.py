
import raop.pipeline as pipeline
import raop.model.ml as model
import numpy as np
import os

trainFile = 'resources/2-train-preprocessed-keys-added.json'

modelOutpath = 'resources/models/'

#fetch features and requestor results (i.e. X's and Y's)
features, pizzas = pipeline.getFeatures(trainFile)
features = np.array(features) 
pizzas = np.array(pizzas)

#my model details
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
modelName = "GaussianNaiveBayes"
directoryName = "GaussianNaiveBayes"
description = "Steven Zimmerman - March 3rd 2015 - Gaussian Naive Bayes"

#create model & metrics  TODO:add precision/recall/F1
modelObj = model.MLmodel(classifier, features, pizzas)
modelObj.crossValidate()
cvStats = modelObj.cv_accuracy
cvStats = cvStats.tolist()
modelObj.fitModel()

#output model to directory
dirPath = modelOutpath + directoryName
if not os.path.exists(dirPath):
    os.makedirs(dirPath)
    

fullOutPath = dirPath + '/' + modelName
modelObj.saveModel(fullOutPath)

#####save report to directory#####
reportFile = fullOutPath + '.report'
oFile = open(reportFile,'w')


#cross-valid stats
oFile.write('Cross Validation Stats\n')
oFile.write('----------------------\n')
i = 1
foldTotal = 0.0
for fold in cvStats:
    oFile.write("Fold-" + str(i) + " = " + str(fold) + '\n')
    i += 1
    foldTotal += fold

oFile.write('----------------------\n')
oFile.write("Average = " + str(foldTotal/(i-1)))


oFile.close()
