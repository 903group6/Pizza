import re

class FeatureExtract(object):
    def  __init__(self):
        self.evidence = None
        self.statusKarma = None
	self.statusAccAge = None
	self.statusPrevAct = None
	    self.narrativeCountMoney1 = 0
        self.narrativeCountMoney2 = 0
        self.narrativeCountJob = 0
        self.narrativeCountFamily = 0
	self.findReciprocity = 0

    def findEvidence(self,request_text_edit_aware):
	'''Find reddit post with image/proof
	   On roap, users often post thier photo/video 
           as an evidence of thier condition/existence
	   this module uses regex library (re) to parse
	   url from imgur and youtube
           As well as url with jpg and png file 		
	   1 - evidence is found
           0 - evidence is not found
        '''

	regex = re.compile(r"([Hh]ttp(\S)+((imgur)|(youtube))(\w|\W)+)|([Hh]ttp(\S)+((jpe?g)|(png)))")
	match = re.search(regex,request_text_edit_aware)	
	if match: 	
            self.evidence = 1
	else :
	    self.evidence = 0

    def evalStatus(self,requester_upvotes_minus_downvotes_at_request, \
	requester_account_age_in_days_at_request, requester_number_of_comments_in_raop_at_request,\
	requester_number_of_posts_on_raop_at_request):
	'''evaluate the status of the requester
	   statusKarma = overall karma point at request time
	   statusAccAge = account age at request time
	   statisPrevAct 1-->previously active on raop, else 0
	'''
	self.statusKarma = requester_upvotes_minus_downvotes_at_request
	self.statusAccAge = requester_account_age_in_days_at_request
	self.statusPrevAct = 1 if requester_number_of_comments_in_raop_at_request + \
	requester_number_of_posts_on_raop_at_request > 0 else 0
	
   def identifyNarratives(self,request_text_edit_aware):
        money1_regex = re.compile(r"(week|ramen|paycheck|work|couple|rice|check|pizza|grocery|rent|anyone|favor|someone|bill|money)")
        money2_regex = re.compile(r"(food|money|house|bill|rent|stamp|month|today|parent|help|pizza|someone|anything|mom|anyone)")
        job_regex = re.compile(r"(job|month|rent|year|interview|bill|luck|school|pizza|paycheck|unemployment|money|ramen|end|check)")
        family_regex = re.compile(r"(tonight|night|today|tomorrow|someone|anyone|friday|dinner|something|account|family|bank|anything|home|work)")
        
        money1_match = re.findall(money1_regex,request_text_edit_aware)
        money2_match = re.findall(money2_regex,request_text_edit_aware)
        job_match = re.findall(job_regex,request_text_edit_aware)
        family_match = re.findall(family_regex,request_text_edit_aware)
        
        self.narrativeCountMoney1 = len(money1_match)
        self.narrativeCountMoney2 = len(money2_match)
        self.narrativeCountJob = len(job_match)
        self.narrativeCountFamily = len(family_match)

   def identifyReciprocity(self, request_text_edit_aware):
	reciprocity_regex = re.compile(r"(pay it forward|([return the favor|([reciprocate))")
	match = re.findall(reciprocity_regex,request_text_edit_aware
	self.findReciprocity = len(match)

    def countWord(self,tokens):
        self.wordNum = len(tokens)
