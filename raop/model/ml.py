import numpy as np
import pickle
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

class MLmodel(object):
    #TODO: Docstring for class
    '''
 
    '''
    def __init__(self, ml_algorithm,X_set,Y_set):
        self.confusionMatrix = None
        self.evalResult = None
        self.cv_accuracy = None
        self.ml_algorithm = ml_algorithm
        self.model = ml_algorithm
        self.modelFileName = None
        self.X_set = X_set
        self.Y_set = Y_set

    def saveModel(self, filename):
        '''Save model as binary files, these files will be necessary
        for future predictions on unseen data'''
        joblib.dump(self.model,filename)


    def crossValidate(self):
         '''Get cross-validation statitics for model
         10-Fold Cross Validation is used'''
         self.cv_accuracy = cross_validation.cross_val_score(\
         self.ml_algorithm, self.X_set, self.Y_set, cv=10)
         

    
    def evaluationResult(self,y_pred):
        '''Get evaluation results e.g. Precision, Recall and F1 scores'''
        self.evalResult = classification_report(self.Y_set,y_pred)
        #TODO: Consider if we want confusion matrix... what value does this add?
        self.confusionMatrix = confusion_matrix(self.Y_set,y_pred)
        
    def fitModel(self):
        '''Given ml_algorithm, build model with X's and Y's and store model'''
        classifier = self.ml_algorithm
        classifier.fit(self.X_set , self.Y_set)
        self.model = classifier
       
  
