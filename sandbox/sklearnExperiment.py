import raop.helper as helper
import raop.pipeline as pipeline
from sklearn.naive_bayes import GaussianNB

keysInFileName = "resources/train.json"
keysOutFileName = "output.json"
inFileName = "output.json"
outFileName = "output2.json"


pipeline.removeNonNeededKeys(keysInFileName,keysOutFileName)
pipeline.addPreprocessedKeyVals(inFileName,outFileName)

actualX, actualY = pipeline.getFeatures(outFileName)

gnb = GaussianNB()
y_pred = gnb.fit(actualX, actualY).predict(actualX)
