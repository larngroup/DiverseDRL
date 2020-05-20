 # -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:25:29 2019

@author: Tiago
"""
import numpy as np
import os
import tensorflow as tf
from keras import backend as K
from utils import *
from dnnQSAR import Model
from alternativeQSAR import build_models
from tokens import tokens_table
from prediction import Predictor
from grid_search_models import grid_search
from tensorflow.python.framework import ops

os.environ["CUDA_VISIBLE_DEVICES"]="0"
session = tf.compat.v1.Session()
K.set_session(session)

#from tensorflow.python.client import device_lib 
#print(device_lib.list_local_devices())

config_file = 'configPredictor.json' # Name of the configuration file 
property_identifier = 'kor' # 'logP', jak2 or 'kor'
model_type = 'dnn' # 'dnn', 'SVM', 'RF', or 'KNN'
descriptor = 'ECFP' # The type of descriptor can be 'SMILES' or 'ECFP'. If we want to use 
#rnn architecture we use SMILES. Conversely, if we want to use a fully connected architecture, we use ECFP descriptors. 
searchParameters = False # True (gridSearch) or False (train with the optimal parameters)
    
def main():
    """
    Main routine: Script that evokes all the necessary routines 
    """
    
    # load model configurations
    config = load_config(config_file,property_identifier)
    directories([config.checkpoint_dir])
    
    # Load the table of possible tokens
    token_table = tokens_table().table
    
    # Read and extract smiles and labels from the csv file
    smiles_raw,labels_raw = reading_csv(config,property_identifier)

    if model_type != 'dnn' or descriptor == 'ECFP':
        # Transformation of data from SMILES strings to ECFP
        data_ecfp = SMILES_2_ECFP(smiles_raw)
    else:
        
        # Padd each SMILES string with spaces until reaching the size of the largest molecule
        smiles_padded,padd = pad_seq(smiles_raw,token_table,0)
        config.paddSize = padd
                
        # Compute the dictionary that makes the correspondence between each token and unique integers
        tokenDict = smilesDict(token_table)
    
    	# Tokenize - transform the SMILES strings into lists of tokens 
        tokens = tokenize(smiles_padded,token_table)   

        # Transforms each token to the respective integer, according to the previously computed dictionary
        smiles_int = smiles2idx(tokens,tokenDict)


    if searchParameters:
    	# Split data into training, validation and testing sets.
        data = data_division(config,smiles_int,labels_raw,False)
        
        # Normalize the label 
        data,data_aux = normalize(data)
        
        # Drop Rate
        drop_rate = [0.1,0.3,0.5]
        # Batch size
        batch_size = [16,32]
        # Learning Rate
        learning_rate=[0.001,0.0001,0.01]
        # Number of cells
        number_units = [64,128,256]
        # Activation function
        activation = ['linear','softmax', 'relu']
        # Memory cell
        rnn = ['LSTM','GRU']
        epochs = [100]
        counter = 0
        for dr in drop_rate:
            for bs in batch_size:
                for lr in learning_rate:
                    for nu in number_units:
                        for act in activation:
                            for nn in rnn:
                                for ep in epochs:
                                    
                                    param_identifier = [str(dr)+"_"+str(bs)+"_"+str(lr)+"_"+
                                                    str(nu)+"_"+nn+"_"+act+"_"+str(ep)]
                                    counter += 1
                                    if counter > 264:
                                        print("\nTesting this parameters: ") 
                                        print(param_identifier)
                                        config.dropout = dr
                                        config.batch_size = bs
                                        config.lr = lr 
                                        config.n_units = nu
                                        config.activation_rnn = act
                                        config.rnn = nn
                                        Model(config,data,searchParameters,descriptor)
 
    
    if model_type == 'dnn' and descriptor == 'SMILES':
        # Data splitting and Cross-Validation for the SMILES-based neural network
        data_rnn_smiles = data_division(config,smiles_int,labels_raw,True,model_type,descriptor)
        x_test = data_rnn_smiles[2]
        y_test = data_rnn_smiles[3]
        data_cv = cv_split(data_rnn_smiles,config)
    
    elif model_type == 'dnn' and descriptor == 'ECFP':
    	# Data splitting and Cross-Validation for the ECFP-based neural network
        data_rnn_ecfp = data_division(config,data_ecfp,labels_raw,True,model_type,descriptor)
        x_test = data_rnn_ecfp[2]
        y_test = data_rnn_ecfp[3]
        data_cv = cv_split(data_rnn_ecfp,config)
    else:
    	# Data splitting, cross-validation and grid-search for the other standard QSAR models        
        data_otherQsar = data_division(config,data_ecfp,labels_raw,True,model_type,descriptor)    
        x_test = data_otherQsar[2]
        y_test = data_otherQsar[3]
        data_cv = cv_split(data_otherQsar,config)
        best_params = grid_search(data_otherQsar,model_type)

    i = 0
    utils = []
    metrics = []
    for split in data_cv:
        print('Cross validation, fold number ' + str(i) + ' in progress...')
        data_i = []
        train, val = split
        
        if model_type != 'dnn' or descriptor == 'ECFP':
            X_train = data_ecfp.iloc[train,:]
            y_train = np.array(labels_raw)[train]
            X_val = data_ecfp.iloc[val,:]
            y_val = np.array(labels_raw)[val]
            y_train = y_train.reshape(-1,1)
            y_val= y_val.reshape(-1,1)
            
        else:
            X_train = smiles_int[train]
            y_train = np.array(labels_raw)[train]
            X_val = smiles_int[val]
            y_val = np.array(labels_raw)[val]
            y_train = y_train.reshape(-1,1)
            y_val= y_val.reshape(-1,1)

            

        data_i.append(X_train)
        data_i.append(y_train)
        data_i.append(x_test)
        data_i.append(y_test)
        data_i.append(X_val)
        data_i.append(y_val)
   
        data_i,data_aux = normalize(data_i)
         
        utils.append(data_aux)
                
        config.model_name = "model" + str(i)
        
        if model_type == 'dnn':
            Model(config,data_i,False,descriptor)
        else:
            build_models(data_i,model_type,config,best_params)

        i+=1
    
    # Model's evaluation with two SMILES strings 
    predictor= Predictor(config,token_table,model_type,descriptor)
    list_ss = ["CC(=O)Nc1cccc(C2(C)CCN(CCc3ccccc3)CC2C)c1","CN1CCC23CCCCC2C1Cc1ccc(O)cc13"] #5.96 e 8.64
    prediction = predictor.predict(list_ss,utils)
    print(prediction)
    
    # Model's evaluation with the test set
    metrics = predictor.evaluator(data_i)
    print("\n\nMean_squared_error: ",metrics[0],"\nR_square: ", metrics[1], "\nRoot mean square: ",metrics[2], "\nCCC: ",metrics[3])
   
if __name__ == '__main__': 
    
    start = time.time()
    print("start time:", start)
    main()
    end = time.time()
    print("\n\n Finish! Time is:", end - start)