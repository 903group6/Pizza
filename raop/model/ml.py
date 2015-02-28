import numpy as np
import pickle
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

class MLmodel(object):
    '''
 
    '''
    def __init__(self, ml_algorithm,X_set,Y_set):
        self.confusionMatrix = None
        self.evalResult = None
        self.cv_accuracy = None
        self.ml_algorithm = ml_algorithm
        self.modelFileName = None
        self.X_set = X_set
        self.Y_set = Y_set

    def saveModel(self, filename):
        joblib.dump(self.ml_algorithm,filename)

    #def cvResult(self):
        #return 0

    def crossValidate(self):
         '''10-Fold Cross Validation'''
         self.cv_accuracy = cross_validation.cross_val_score(\
         self.ml_algorithm, self.X_set, self.Y_set, cv=10)
         

    
    def evaluationResult(self,y_pred):
	self.evalResult = classification_report(self.Y_set,y_pred)
        self.confusionMatrix = confusion_matrix(self.Y_set,y_pred)
       
  
