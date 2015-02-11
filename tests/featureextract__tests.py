from nose.tools import *
import raop.featureextract.featureextract as featureextract

def testEvidence():
    '''Test for find evidence function'''
    testTextEvi_true = "This is the state of my refrigerator: [Linky](http://i.imgur.com/JRvYJ5s.jpg)"
    testTextEvi_false = "http://www.4chan.org was not my favorite website "
  
    featObj = featureextract.FeatureExtract()
    featObj.findEvidence(testTextEvi_true)
    assert_equal(featObj.evidence,1)
    featObj.findEvidence(testTextEvi_false)
    assert_equal(featObj.evidence,0)
     
def testStatus():
    '''Test for status evaluation function'''
    featObj = featureextract.FeatureExtract()
    featObj.evalStatus(56,365.123,0,0)
    assert_equal(featObj.statusKarma,56)
    assert_equal(featObj.statusAccAge,365.123)
    assert_equal(featObj.statusPrevAct,0)
    featObj.evalStatus(-56,1.0000,3,6)
    assert_equal(featObj.statusKarma,-56)
    assert_equal(featObj.statusAccAge,1.0000)
    assert_equal(featObj.statusPrevAct,1)
