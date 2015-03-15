import produceModelsStatsandSubmission as masterPipe


#data sets
trainFile = 'resources/2-train-preprocessed-keys-added.json'
testFile = 'resources/test-preprocessed-keys-added.json'

#root for all models, results and submission files
modelOutpath = 'resources/models/'


#CREATE model and produce evaluation results
# The fields  below should be updated to reflect the model and additional features included
import sklearn.svm as svm 

classifier = svm.SVC(kernel='linear',class_weight = 'auto')
modelName = "SVM-Norm-linear-CLauto-Add-JJ-RB-NN-VB-CansBinary"
directoryName = "SVM"
description = "8th March 2015 "


masterPipe.buildModels(classifier, modelName, directoryName, modelOutpath,description, [1,2,3,4],trainFile, testFile)
