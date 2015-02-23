import json
from nltk.tokenize import RegexpTokenizer

money1 = ["week", "ramen", "paycheck", "work", "couple", "rice",
"check", "pizza", "grocery","rent", "anyone", "favor",
"someone","bill","money"]
	
money2 = ["food", "money", "house", "bill", "rent", "stamp", "month",
"today", "parent", "help", "pizza", "someone", "anything"
"mom", "anyone"]

job =  ["job", "month", "rent", "year", "interview", "bill", "luck",
"school", "pizza", "paycheck", "unemployment",
"money", "ramen", "end", "check"]

family = ["tonight", "night", "today", "tomorrow", "someone",
"anyone", "friday", "dinner", "something", "account",
"family", "bank", "anything", "home", "work"]       #also includes time

instanceCounts = {"countMoney1": 0,"countMoney2": 0,"countJob": 0,"countFamily": 0}

jsonFile=open("C:\\Users\\Tom\\Downloads\\train.json\\train.json").read()
trainData = json.loads(jsonFile)

tokenizer = RegexpTokenizer('[A-Za-z]+') 

for record in trainData:
    instanceCounts["countMoney1"] = 0
    instanceCounts["countMoney2"] = 0
    instanceCounts["countJob"] = 0
    instanceCounts["countFamily"] = 0
    recordContents = record['request_text_edit_aware']
    recordContents = tokenizer.tokenize(recordContents)
    for line in recordContents:
        if line in money1:
            instanceCounts["countMoney1"] += 1
        if line in money2:
            instanceCounts["countMoney2"] += 1
        if line in job:
            instanceCounts["countJob"] += 1
        if line in family:
            instanceCounts["countFamily"] += 1
    record.update(instanceCounts)

outFile = open("trainOutput.json","w")
json.dump(trainData,outFile,sort_keys = True, indent = 4)
outFile.close
    
