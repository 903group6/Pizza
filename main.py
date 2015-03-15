import produceModelsStatsandSubmission as masterPipe


#data sets
trainFile = 'resources/2-train-preprocessed-keys-added.json'
testFile = 'resources/test-preprocessed-keys-added.json'

#root for all models, results and submission files
modelOutpath = 'resources/models/'
reportPath = 'resources/reports/'
submissionsPath = 'resources/submissions/'

#global model details
baseFeatDesc = 'B-ALL'
#baseFeatList = [1,2,3,4,5,6,7,8,9,10,11,12]  #12 Total
baseFeatList = [1,2,3,4,5,6,7,8,9,10,11,12]  #12 Total
addFeatDesc = 'A-none'
addFeatList = []  #e.g. [1,2,4]


#CREATE model and produce evaluation results
# The fields  below should be updated to reflect the model and additional features included
import sklearn.svm as svm 

classifier = svm.SVC(kernel='linear',class_weight = 'auto')
modelName = "SVM-Norm-linear-CLauto-" + baseFeatDesc + '-' + addFeatDesc
directoryName = "SVM"
description = "15th March 2015 "

masterPipe.buildModels(classifier, modelName, directoryName, modelOutpath, \
                        description,baseFeatList,addFeatList,trainFile, testFile,\
                        reportPath, submissionsPath)
 
###basic SVC                     
#classifier = svm.SVC()
#modelName = "SVM-Norm-" + baseFeatDesc + '-' + addFeatDesc
#directoryName = "SVM"
#description = "15th March 2015 "

#masterPipe.buildModels(classifier, modelName, directoryName, modelOutpath, \
#                        description,baseFeatList,addFeatList,trainFile, testFile,\
#                        reportPath, submissionsPath)                     
                        
#gaussian
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
modelName = "GaussianNaiveBayes-Norm-" + baseFeatDesc + '-' + addFeatDesc
directoryName = "GaussianNaiveBayes"
description = "Steven Zimmerman - 15th March 2015  - Gaussian Naive Bayes"
masterPipe.buildModels(classifier, modelName, directoryName, modelOutpath, \
                        description,baseFeatList,addFeatList,trainFile, testFile,\
                        reportPath, submissionsPath)   


#RFC
#from sklearn.ensemble import RandomForestClassifier
#classifier =RandomForestClassifier()
#modelName = "RandomForestClassifier-Norm-" + baseFeatDesc + '-' + addFeatDesc
#directoryName = "RandomForestClassifier"
#description = "Can Udomcharoenchaikit - 15th March 2015 - Random Forest"
#masterPipe.buildModels(classifier, modelName, directoryName, modelOutpath, \
#                        description,baseFeatList,addFeatList,trainFile, testFile,\
#                        reportPath, submissionsPath)  
