import raop.helper as helper
from sklearn.naive_bayes import GaussianNB

'''
Inputs ==>
1) JSON file with dictionary containing features and target classifications
2) Key Names for features and target class

TODO:
 Outputs  ==>
1) Dictionary in required format for scikit learn e.g. {'data':[[0,0,2,1,4],[0,2,1,5,3]]
																												'targets':{[0,1]

'''

featureKeys = ["requester_account_age_in_days_at_request",
							"requester_days_since_first_post_on_raop_at_request",
							"requester_number_of_comments_at_request",
							"requester_number_of_comments_in_raop_at_request",
							"requester_number_of_posts_at_request",
							"requester_number_of_posts_on_raop_at_request",
							"requester_number_of_subreddits_at_request",
							"requester_upvotes_minus_downvotes_at_request",
							"requester_upvotes_plus_downvotes_at_request"]
							
targetKey = "requester_received_pizza"


##LOAD TRAINING DATA
list = helper.loadJSONfromFile("resources/1-train-fields-removed.json")

#feature and target vectors
data = []
targets = []

#get features and target vectors
for dict in list:
	currentFeatures = []
	targets.append(dict.get( targetKey ))
	for feature in featureKeys:
		currentFeatures.append(dict.get(feature))
	data.append(currentFeatures)
	

#Run Naive Bayes on data and target vectors (these are modified versions of code from scikit learn tutorial)
gnb = GaussianNB()
y_pred = gnb.fit(data, targets).predict(data)

#print the results
print("Number of mislabeled points out of a total %d points : %d"
      % (len(data),(targets != y_pred).sum()))