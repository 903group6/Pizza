import raop.helper as helper
import raop.preprocess.preprocess as preproc
import raop.featureextract.featureextract as featureextract
import raop.model.ml as model
import numpy as np
import os
import nltk
from sklearn.cross_validation import train_test_split as train_valid_split

#Step 1	- Remove desired keys from each dictionary
def removeNonNeededKeys(inputJSONfile,outputJSONfile):
	'''Removes keys from training file that are not needed.  
	The keys listed below are not in the test data and therefore not necessary in training data either.
	These fields are removed for readibility
	usage: removeNonNeededKeys("resources/train.json","resources/1-train-fields-removed.json")'''
	testInput = "resources/train.json"
	testOutput = "resources/1-train-fields-removed.json"
	keysToDrop = ["number_of_downvotes_of_request_at_retrieval", 
					"number_of_upvotes_of_request_at_retrieval", 
					"post_was_edited", 
					"request_number_of_comments_at_retrieval", 
					"request_text", 
					"requester_account_age_in_days_at_retrieval", 
					"requester_days_since_first_post_on_raop_at_retrieval", 
					"requester_number_of_comments_at_retrieval", 
					"requester_number_of_comments_in_raop_at_retrieval", 
					"requester_number_of_posts_at_retrieval", 
					"requester_number_of_posts_on_raop_at_retrieval", 
					"requester_user_flair",
					"requester_upvotes_minus_downvotes_at_retrieval", 
					"requester_upvotes_plus_downvotes_at_retrieval",]

	list = helper.loadJSONfromFile(inputJSONfile)
	for dict in list:				
		for key in keysToDrop:
				dict.pop(key, None)	

	helper.dumpJSONtoFile(outputJSONfile, list)

###########################

#Step 2 - Add POS tags, tokens, etc to each dictionary
def addPreprocessedKeyVals(inputJSONfile,outputJSONfile):
	'''Loads json file to list --> creates object for each dictionary in list
	Then preprocesses the text data in dictionary (e.g. POS tags)
	Then creates new key value pairs with these processed fields
	usage: addPreprocessedKeyVals("resources/1-train-fields-removed.json","resources/2-train-preprocessed-keys-added.json")'''
	list = helper.loadJSONfromFile(inputJSONfile)
	#count = 1
	for dict in list:
		preProcObj = preproc.Preprocess()
		preProcObj.setDictionary(dict)
		preProcObj.concatenate("request_title", "request_text_edit_aware")
		preProcObj.sentSeg(preProcObj.concatText)
		preProcObj.tokenize(preProcObj.concatText)
		preProcObj.posTag(preProcObj.tokenizedText)
		preProcObj.normalisation(preProcObj.tokenizedText)
		dict["added_Title_+_Request"] = preProcObj.concatText
		dict["added_segmented_sentences"] = preProcObj.sentSegmentedText
		dict["added_tokens"] = preProcObj.tokenizedText
		dict["added_POStags"] = preProcObj.POS_TaggedText
		dict["added_normalised_text"] = preProcObj.normalisedText
		#print count
		#count += 1
	helper.dumpJSONtoFile(outputJSONfile, list)


###########################

#Step 3 - Extract Features / Create Feature Vectors
def getFeatures(inputJSONfile, isTest):
    '''Loads Json(output from step 2) file to list --> creates object for 
       each dictionary in list. Then extract features from each dictionary,
       and keep it in a feature vector. Once feature extraction is completed,
       normalise the feature vectors.
       '''
    thelist = helper.loadJSONfromFile(inputJSONfile)
    featObj = featureextract.FeatureExtract()
    X_set = []
    Y_set = []
    featObj.getMinTime(thelist)

    for dict in thelist:
        temp_feat = []
        featObj.findEvidence(dict["added_Title_+_Request"])
        featObj.evalStatus(dict["requester_upvotes_minus_downvotes_at_request"],\
        dict["requester_account_age_in_days_at_request"],\
        dict["requester_number_of_comments_in_raop_at_request"],\
        dict["requester_number_of_posts_on_raop_at_request"])
        temp_feat.append(featObj.evidence)
        temp_feat.append(featObj.statusKarma)
        temp_feat.append(featObj.statusAccAge)
        temp_feat.append(featObj.statusPrevAct)
        featObj.identifyNarratives(dict["added_Title_+_Request"])
        temp_feat.append(featObj.narrativeCountMoney1)
        temp_feat.append(featObj.narrativeCountMoney2)
        temp_feat.append(featObj.narrativeCountJob)
        temp_feat.append(featObj.narrativeCountFamily)
         
        featObj.identifyReciprocity(dict["added_Title_+_Request"])
        featObj.countWord(dict["added_tokens"])
        temp_feat.append(featObj.findReciprocity)
        temp_feat.append(featObj.wordNum)
        featObj.getTime(dict["unix_timestamp_of_request"])
        featObj.getFirstHalf(dict["unix_timestamp_of_request"])
        temp_feat.append(featObj.time)
        temp_feat.append(featObj.firstHalf)
        X_set.append(temp_feat)
        if isTest==0:
            Y_set.append(dict["requester_received_pizza"])
        #TO DO:Normalisation/Vectorization
    
    #TODO: Change this to return numpy arrays??? it is required for models
    if isTest ==0:
        return np.array(X_set), np.array(Y_set)
    else:
        return np.array(X_set)
    
#####################
#Step 4 - Generic pipeline component to build model, save model, and output
#         evaluation metrics
def modelPipeline(classifier, X_set, Y_set,\
    modelOutpath, modelName, directoryName, description):
    '''
    INPUTS: 
    classifier = any sklearn classifier ex: GaussianNB()
    X_set = all features in numpy array format
    Y_set = all classifications in numpy array format (e.g. pizza True/False)
    
    modelOutpath = directory location of all models
    modelName = the name of model (perhaps a unique name)
    directoryName = the name of new directory to create your model
    description = full description of model (e.g. who ran it, date, features included)
    
    
    OUTPUTS:
    - Model binary files to specified directory
    - Report with details on model (e.g. Metrics such as F1/Precision, 
    cross-validation stats)
     
    '''
    #create Validation model & cross-valid metrics  
    #split training set into 80/20 split.  This will be useful for evaluation metrics
    X_train, X_valid , Y_train, Y_valid = train_valid_split(X_set,Y_set,test_size=0.2, random_state=42)
    modelObj = model.MLmodel(classifier, X_train, Y_train)
    modelObj.crossValidate()
    cvStats = modelObj.cv_accuracy
    cvStats = cvStats.tolist()
    modelObj.fitModel()

    #Create validation model predictions & calculate precision/recall/F1
    y_preds = modelObj.model.predict(X_valid)  #get predictions for model
    modelObj.evaluationResult(y_preds,Y_valid)

    #set output model to directory
    dirPath = modelOutpath + directoryName
    fullOutPath = dirPath + '/' + modelName
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)


    #####save report to directory#####
    reportFile = fullOutPath + '.report'
    oFile = open(reportFile,'w')

    #report information
    oFile.write('Model Name: ' + modelName + '\n')
    oFile.write('Model Description: ' + description + '\n')
    oFile.write('Model Feautures: ' + 'BASELINE FEATURES' + '\n\n')

    #prec/recall/f-1
    oFile.write('Model Evaluation Metrics\n')
    oFile.write('----------------------\n\n')
    oFile.write(modelObj.evalResult)
    oFile.write('\n\n----------------------\n\n')

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
    oFile.write("Average = " + str(foldTotal/(i-1)) + '\n')


    oFile.close()

    #RELOAD DATA and CREATE FINAL MODEL
    #reload training data and create model on 100% of training set
    modelObj = model.MLmodel(classifier, X_set, Y_set)
    modelObj.fitModel()
    
    #save final model
    modelObj.saveModel(fullOutPath)

