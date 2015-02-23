from nose.tools import *
import raop.helper as helper
import raop.pipeline as pipeline

keysInFileName = "test-data/oneJSONentry.json"
keysOutFileName = "test-data/output-part1-keys-removed-oneJSONentry.json"
inFileName = "test-data/utf-encoding-test.json"
outFileName = "test-data/output-part2-utf-encoding-test.json"

def test_part1_remove_keys():
	'''Test for part 1 of pipeline. The keys listed in function must be removed'''
	expectedOutput = helper.loadJSONfromFile(keysOutFileName)
	pipeline.removeNonNeededKeys(keysInFileName,keysOutFileName)
	actualOutput = helper.loadJSONfromFile(keysOutFileName)
	assert_equal(expectedOutput,actualOutput)


def test_part2_withUTF():
	'''Test for part 2 of pipeline. The text string must be encoded and decoded 
	therefore this is tested too'''
	expectedOutput = helper.loadJSONfromFile(outFileName)
	pipeline.addPreprocessedKeyVals(inFileName,outFileName)
	actualOutput = helper.loadJSONfromFile(outFileName)
	assert_equal(expectedOutput,actualOutput)

def test_part3_getFeatures():
	'''Test for part 3 of pipeline. 
        Extract features from JSON file and Create a feature vectors for each instance'''
        expectedX=[[0,616,294.1380208333333,0,3,5,4,4]]
	expectedY=[False]
	actualX, actualY = pipeline.getFeatures(outFileName)
        assert_equal(expectedX,actualX)
	assert_equal(expectedY,actualY)
