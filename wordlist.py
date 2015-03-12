import raop.helper as helper
import os
import csv

def cleanUpWord(currentWord):
    #remove periods at end and white space in front and back
    currentWord = currentWord.strip()  #remove white space
    currentWord = currentWord.strip('.') #remove periods at end and front  


def getLocalWordList(listOfWords):
    listToReturn = []
    for word in listOfWords:
        cleanWord = word.strip().strip('.')  #remove whitespace and period at front and back
        if cleanWord in listToReturn:
            continue
        else:
            listToReturn.append(cleanWord)
    
    return listToReturn


def getGlobalWordList(inputJSONfile):
    theDicts = helper.loadJSONfromFile(inputJSONfile)
    
    dictToReturn = {}
    for dictionary in theDicts:
        gotPizzaBool = dictionary["requester_received_pizza"]
        currentUniqueWords = getLocalWordList(dictionary["added_normalised_text"])
        for word in currentUniqueWords:
            if word in dictToReturn:
                trfaTotal = dictToReturn[word]
                dictToReturn.pop(word)
                if gotPizzaBool:
                    trfaTotal = [trfaTotal[0] + 1,trfaTotal[1]]
                else:
                    trfaTotal = [trfaTotal[0],trfaTotal[1] + 1]
                dictToReturn[word] = trfaTotal
            else:
                trfaTotal = []
                if gotPizzaBool:
                    trfaTotal = [1,0]
                else:
                    trfaTotal = [0,1]
                dictToReturn[word] = trfaTotal
    return dictToReturn
    

def outputToCSV(wordValsDict,outFile):
    with open(outFile, 'wb') as csvfile:
        csvWriter = csv.writer(csvfile)        
        for key in wordValsDict:
            vals = wordValsDict[key]
            #write out word,true-count,false-count
            csvWriter.writerow([key.encode('utf-8'),str(vals[0]),str(vals[1])])
    
dictOfWordsVals = getGlobalWordList(inputJSON)
outputToCSV(dictOfWordsVals,outputCSV)

inputJSON = 'sandbox/wordlists/preproc.json'
outputCSV = 'sandbox/wordlists/actual.csv'

trainJSON = 'resources/2-train-preprocessed-keys-added.json'
trainCSV = 'resources/wordTrueFalseCounts.csv'
dictOfWordsVals = getGlobalWordList(trainJSON)
outputToCSV(dictOfWordsVals,trainCSV)