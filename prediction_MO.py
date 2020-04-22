# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:46:42 2019

@author: Tiago
"""
import tensorflow as tf 
from tensorflow.keras.models import model_from_json
import sklearn.metrics, math
import csv
import numpy as np
import os
from utils import *
from tokens import tokens_table

class Predictor(object):
    def __init__(self, config, tokens,labels_jak2,labels_logP):
        super(Predictor, self).__init__()
        self.labels_jak2 = labels_jak2
        self.labels_logP = labels_logP
        self.tokens = tokens
        self.config = config
        
        loaded_models_jak2= []
        loaded_models_logP = []
        for i in range(5):
            
            json_file_jak2 = open("predictor_models_jak2\\model"+str(i)+".json", 'r')
            json_file_logP = open("predictor_models_logP\\model"+str(i)+".json", 'r')
            
            loaded_model_json_jak2 = json_file_jak2.read()
            loaded_model_json_logP = json_file_logP.read()
            
            json_file_jak2.close()
            json_file_logP.close()
            
            loaded_model_jak2 = model_from_json(loaded_model_json_jak2)
            loaded_model_logP = model_from_json(loaded_model_json_logP)
            
            # load weights into new model
            loaded_model_jak2.load_weights("predictor_models_jak2\\model"+str(i)+".h5")
            loaded_model_logP.load_weights("predictor_models_logP\\model"+str(i)+".h5")
            
            print("Models " + str(i) + " loaded from disk!")
            loaded_models_jak2.append(loaded_model_jak2)
            loaded_models_logP.append(loaded_model_logP)
        
        self.loaded_models_jak2 = loaded_models_jak2
        self.loaded_models_logP = loaded_models_logP
        
    def predict(self, smiles,objective):
        
        models = []
        labels = []
        if objective == 'jak2':
            models = self.loaded_models_jak2
            labels = self.labels_jak2
        elif objective == 'logP':
             models = self.loaded_models_logP
             labels = self.labels_logP
        
        smiles_padded,kl = pad_seq(smiles,self.tokens,self.config.paddSize)
        
        d = smilesDict(self.tokens)
  
        smiles_int = smiles2idx(smiles_padded,d)
        
        prediction = []
            
        for m in range(len(models)):
                prediction.append(models[m].predict(smiles_int))
                
        prediction = np.array(prediction).reshape(len(models), -1)
        
        prediction = denormalization(prediction,labels)
                
        prediction = np.mean(prediction, axis = 0)
     
        return prediction
            
