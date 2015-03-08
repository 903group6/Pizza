import raop.pipeline as pipeline
import raop.model.ml as model
from sklearn.externals import joblib
import numpy as np
import raop.helper as helper
import os

############get scaling paramaters################
trainFile = 'resources/2-train-preprocessed-keys-added.json'

modelOutpath = 'resources/models/'

#fetch features and requestor results (i.e. X's and Y's)
features, pizzas = pipeline.getFeatures(trainFile,0)

#normalize
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(features)

############# end get scaling features #####################

#kaggle test data
testFile = 'resources/test-preprocessed-keys-added.json'

#path and filename info
modelOutpath = 'resources/models/'
#TODO: Fix the naming conventions... should be dirName not modelFile
modelName = "SVM"
modelFile = 'SVM-Norm-linear-CLauto-timeHardCode'
inputModelFileName = modelOutpath + modelName + '/' + modelFile

model = joblib.load(inputModelFileName)

#fetch features  (i.e. X's )
features = pipeline.getFeatures(testFile,1)
listofID = []
testdata = helper.loadJSONfromFile(testFile)
for item in testdata:
    listofID.append(item["request_id"])
    
#scale the features
features = scaler.transform(features)


#get results predcitions and convert to list		
results = model.predict(features)
results_list=results.tolist()


#write results and user details to submission file
outputFile=open('results.csv','w')

outputFile.write("request_id,requester_received_pizza\n")
for counter,id in enumerate(listofID):
    outputFile.write(id+",")
    if results_list[counter]:
        outputFile.write('1\n')
    else:
        outputFile.write('0\n')

outputFile.close()


