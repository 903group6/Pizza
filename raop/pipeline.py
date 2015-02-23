import raop.helper as helper
import raop.preprocess.preprocess as preproc
import raop.featureextract.featureextract as featureextract

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
	count = 1
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
		print count
		count += 1
	helper.dumpJSONtoFile(outputJSONfile, list)


###########################

#Step 2 - Extract Features / Create Feature Vectors
def getFeatures(inputJSONfile):
    '''Loads Json(output from step 2) file to list --> creates object for 
       each dictionary in list. Then extract features from each dictionary,
       and keep it in a feature vector. Once feature extraction is completed,
       normalise the feature vectors.
       '''
    list = helper.loadJSONfromFile(inputJSONfile)
    featObj = featureextract.FeatureExtract()
    X_set = []
    Y_set = []
    for dict in list:
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
    	temp_feat.append(featObj.narrativeCountMoney1)
    	temp_feat.append(featObj.narrativeCountMoney2)
    	temp_feat.append(featObj.narrativeCountJob)
    	temp_feat.append(featObj.narrativeCountFamily)
    	temp_feat.append(featObj.findReciprocity)
    	temp_feat.append(featObj.wordNum)
    	
	X_set.append(temp_feat)
	Y_set.append(dict["requester_received_pizza"])
    #TO DO:Normalisation/Vectorization
    
    return X_set, Y_set

