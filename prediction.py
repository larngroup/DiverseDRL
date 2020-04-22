# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:46:42 2019

@author: Tiago
"""
import tensorflow as tf 
from tensorflow.keras.models import model_from_json
import sklearn.metrics
import numpy as np
from utils import *
from tokens import tokens_table

class Predictor(object):
    def __init__(self, config, tokens,labels,property_identifier):
        """
        Constructor for the Predictor object.
        Parameters
        ----------
        config: Configuration parameters 
        tokens: Table with all possible tokens in SMILES
        labels: True values of Predictor training dataset to perform the 
                denormalization step
        property_identifier: String identifying the property to optimize
        
        Returns
        -------
        This function loads the Predictor models already trained
        """
        super(Predictor, self).__init__()
        self.labels = labels
        self.tokens = tokens
        self.config = config

        loaded_models = []
        if property_identifier == "jak2":
            model_path = "predictor_models_jak2\\model"
        elif property_identifier == "logP":
            model_path = "predictor_models_logP\\model"
            
        for i in range(5):
            
            json_file= open(model_path+str(i)+".json", 'r')
            
            
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)            
            # load weights into new model
            loaded_model.load_weights(model_path+str(i)+".h5")
            print("Models " + str(i) + " loaded from disk!")
            loaded_models.append(loaded_model)
        
        self.loaded_models = loaded_models

        
    def predict(self, smiles):
        """
        This function performs the prediction of the property in study
        Parameters
        ----------
        smiles: List of SMILES strings to perform the prediction      
        Returns
        -------
        Before do the prediction, this function performs the SMILES' padding 
        and tokenization. It also performs the denormalization step and compute 
        the mean value of the prediction of the 5 models.
        """
  
        smiles_padded,kl = pad_seq(smiles,self.tokens,self.config.paddSize)
        
        d = smilesDict(self.tokens)
  
        smiles_int = smiles2idx(smiles_padded,d)
        
        prediction = []
            
        for m in range(len(self.loaded_models)):
                prediction.append(self.loaded_models[m].predict(smiles_int))
                
        prediction = np.array(prediction).reshape(len(self.loaded_models), -1)
        
        prediction = denormalization(prediction,self.labels)
                
        prediction = np.mean(prediction, axis = 0)
     
        return prediction
            
