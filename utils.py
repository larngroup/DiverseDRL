# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 23:22:10 2019

@author: Tiago
"""
import csv
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def reading_csv(config,property_identifier):
    """
    This function loads the labels of the property specified by the identifier.
    ----------
    config: configuration file
    property_identifier: Identifier of the property to optimize.
    
    Returns
    -------
    raw_labels: Returns the labels in a numpy array. 
    """
    
    if property_identifier == "jak2":
        file_path = config.file_path_jak2
    elif property_identifier == "logP":
        file_path = config.file_path_logP
        
    raw_labels = []
        
    with open(file_path, 'r') as csvFile:
        reader = csv.reader(csvFile)
        
        it = iter(reader)
        next(it, None)  # skip first item.    
        for row in it:
            if property_identifier == "jak2":
                raw_labels.append(float(row[1]))
            elif property_identifier == "logP":
                raw_labels.append(float(row[2]))            
    return raw_labels
        

#def get_tokens(smiles):  
#    tokens = []
#    
#    for smile in smiles:
#        for token in smile:
#            if token not in tokens:
#                tokens.append(token)
#    return tokens
           

def smilesDict(tokens):
    """
    This function extracts the dictionary that makes the correspondence between
    each charachter and an integer.
    ----------
    tokens: Set of characters
    
    Returns
    -------
    tokenDict: Returns the dictionary that maps characters to integers
    """
    tokenDict = dict((token, i) for i, token in enumerate(tokens))
    return tokenDict

def pad_seq(smiles,tokens,paddSize):
    """
    This function performs the padding of each SMILE.
    ----------
    smiles: Set of SMILES strings with different sizes;
    tokens: Set of characters;
    paddSize: Integer that specifies the maximum size of the padding    
    
    Returns
    -------
    newSmiles: Returns the padded smiles, all with the same size.
    """
    maxSmile= max(smiles, key=len)
    maxLength = 0
    
    if paddSize != 0:
        maxLength = paddSize
    else:
        maxLength = len(maxSmile) 
  #  maxLength = 167
    for i in range(0,len(smiles)):
        if len(smiles[i]) < maxLength:
            smiles[i] = smiles[i] + tokens[-1]*(maxLength - len(smiles[i]))

    return smiles,maxLength
             
def smiles2idx(smiles,tokenDict):  
    """
    This function transforms each SMILES character to the correspondent integer,
    according the token dictionary.
    ----------
    smiles: Set of SMILES strings with different sizes;
    tokenDict: Dictionary that maps the characters to integers;    
    
    Returns
    -------
    newSmiles: Returns the transformed smiles, with the characters replaced by 
    the numbers. 
    """         
    newSmiles =  np.zeros((len(smiles), len(smiles[0])))
    for i in range(0,len(smiles)):
        for j in range(0,len(smiles[i])):
           
            newSmiles[i,j] = tokenDict[smiles[i][j]]
            
    return newSmiles


def cv_split(smiles,labels,config):
    """
    This function performs the data spliting in 5 folds, training and testing.
    ----------
    smiles: Set of SMILES strings;
    labels: Set of real values of the desired property;
    config: Configuration file;
    
    Returns
    -------
    data: Returns the data indexes splited in 5 folds, and, each fold divided 
    between training and testing sets. 
    """
    cross_validation_split = KFold(n_splits=config.n_splits, shuffle=True)
    data = list(cross_validation_split.split(smiles, labels))
    return data

def scalarization(reward_jak2,reward_logP,scalarMode):
    """
    This function computes a linear scalarization of the two objectives to 
    obtain a unique reward.
    ----------
    reward_jak2: Reward obtained with the optimzation of jak2 property;
    reward_jak2: Reward obtained with the optimzation of logP property;
    scalarMode: Type of scalarization;    
    
    Returns
    -------
    Returns the scalarized reward
    """
    if scalarMode == 'linear':
    
        w1 = 0.5
        w2 = 0.5
        
        return w1*reward_jak2 + w2*reward_logP

def rmse(y_true, y_pred):
    """
    This function implements the root mean squared error measure
    ----------
    y_true: True label   
    y_pred: Model predictions 
    Returns
    -------
    Returns the rmse metric to evaluate regressions
    """
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def mse(y_true, y_pred):
    """
    This function implements the mean squared error measure
    ----------
    y_true: True label   
    y_pred: Model predictions 
    Returns
    -------
    Returns the mse metric to evaluate regressions
    """
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    """
    This function implements the coefficient of determination (R^2) measure
    ----------
    y_true: True label   
    y_pred: Model predictions 
    Returns
    -------
    Returns the R^2 metric to evaluate regressions
    """
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))


#def exp_decay(epoch):
#    initial_lrate = 0.005
#    k = 0.98
#    lrate = initial_lrate * math.exp(-k*epoch)
#    print(lrate)
#    return lrate


def canonical_smiles(smiles, sanitize=True, throw_warning=False):
    """
    Takes list of generated SMILES strings and returns the list of valid SMILES.
    Parameters
    ----------
    smiles: list
        list of SMILES strings to validate
    sanitize: bool (default True)
        parameter specifying whether to sanitize SMILES or not.
            For definition of sanitized SMILES check
            http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol
    throw_warning: bool (default False)
        parameter specifying whether warnings will be thrown if a SMILES is
        invalid
    Returns
    -------
    new_smiles: list of valid SMILES (if it is valid and has <65 characters)
    and NaNs if SMILES string is invalid
    valid: number of valid smiles, regardless of the its size
        
    """
    new_smiles = []
    valid = 0
    for sm in smiles:
        try:
            mol = Chem.MolFromSmiles(sm[0], sanitize=sanitize)
            s = Chem.MolToSmiles(mol)
            
            if len(s) <= 65:
                new_smiles.append(s)
            else:
                new_smiles.append('')
            valid = valid + 1 
        except:
            if throw_warning:
                warnings.warn(sm + ' can not be canonized: invalid '
                                   'SMILES string!', UserWarning)
            new_smiles.append('')
    return new_smiles,valid

def plot_hist(prediction, n_to_generate,valid,property_identifier):
    """
    Function that plots the predictions's distribution of the generated SMILES 
    strings
    Parameters
    ----------
    prediction: list with the desired property predictions.
    n_to_generate: number of generated SMILES.
    valid: number of valid smiles, regardless of the its size.
    property_identifier: String identifying the property 
    """
    prediction = np.array(prediction)
    x_label = ''
    plot_title = '' 
    
    if property_identifier == "jak2":
        
        print("\nProportion of valid SMILES:", len(prediction)/n_to_generate)
        print("Proportion of REAL valid SMILES:", valid/n_to_generate)
        print("Average of IC50: ", np.mean(prediction))
        print("Median of IC50: ", np.median(prediction))
        x_label = "Predicted pIC50"
        plot_title = "Distribution of predicted pIC50 for generated molecules"
        
    elif property_identifier == "logP":
        percentage_in_threshold = np.sum((prediction >= 0.0) & 
                                     (prediction <= 5.0))/len(prediction)
        print("Percentage of predictions within drug-like region:", percentage_in_threshold)
        print("Proportion of valid SMILES:", len(prediction)/n_to_generate)
        print("Proportion of valid SMILES:", valid/n_to_generate)
        print("Average of log_P: ", np.mean(prediction))
        print("Median of log_P: ", np.median(prediction))
        plt.axvline(x=0.0)
        plt.axvline(x=5.0)
        x_label = "Predicted LogP"
        plot_title = "Distribution of predicted LogP for generated molecules"
        
        
    ax = sns.kdeplot(prediction, shade=True,color = 'b')
    ax.set(xlabel=x_label, 
           title=plot_title)
    plt.show()
    
def plot_hist_both(prediction_unb,prediction_b, n_to_generate,valid_unb,valid_b,property_identifier):
    """
    Function that plots the predictions's distribution of the generated SMILES 
    strings, obtained by the unbiased and biased generators.
    Parameters
    ----------
    prediction_unb: list with the desired property predictions of unbiased 
                    generator.
    prediction_unb: list with the desired property predictions of biased 
                    generator.
    n_to_generate: number of generated SMILES.
    valid_unb: number of valid smiles of the unbiased generator, regardless of 
            the its size.
    valid_b: number of valid smiles of the biased generator, regardless of 
            the its size.
    property_identifier: String identifying the property 
    """
    prediction_unb = np.array(prediction_unb)
    prediction_b = np.array(prediction_b)
    
    legend_unb = ''
    legend_b = '' 
    label = ''
    plot_title = ''
    
    if property_identifier == 'jak2':
        legend_unb = 'Unbiased pIC50 values'
        legend_b = 'Biased pIC50 values'
        
        print("Proportion of valid SMILES (UNB,B):", len(prediction_unb)/n_to_generate,len(prediction_b)/n_to_generate )
        print("REAL Proportion of valid SMILES (UNB,B):", valid_unb/n_to_generate,valid_b/n_to_generate )
        print("Average of log_P: (UNB,B)", np.mean(prediction_unb),np.mean(prediction_b))
        print("Median of log_P: (UNB,B)", np.median(prediction_unb),np.median(prediction_b))
    
        label = 'Predicted of pIC50'
        plot_title = 'Distribution of predicted pIC50 for generated molecules'
        
    elif property_identifier == 'logP':
        legend_unb = 'Unbiased logP values'
        legend_b = 'Biased logP values'
        
        percentage_in_threshold_unb = np.sum((prediction_unb >= 0.0) & 
                                            (prediction_unb <= 5.0))/len(prediction_unb)
        percentage_in_threshold_b = np.sum((prediction_b >= 0.0) & 
                                 (prediction_b <= 5.0))/len(prediction_b)
        print("% of predictions within drug-like region (UNB,B):", 
          percentage_in_threshold_unb,percentage_in_threshold_b)
        print("Proportion of valid SMILES (UNB,B):", len(prediction_unb)/n_to_generate,len(prediction_b)/n_to_generate )
        print("REAL Proportion of valid SMILES (UNB,B):", valid_unb/n_to_generate,valid_b/n_to_generate )
        print("Average of log_P: (UNB,B)", np.mean(prediction_unb),np.mean(prediction_b))
        print("Median of log_P: (UNB,B)", np.median(prediction_unb),np.median(prediction_b))
    
        label = 'Predicted logP'
        plot_title = 'Distribution of predicted LogP for generated molecules'
        plt.axvline(x=0.0)
        plt.axvline(x=5.0)
        
    v1 = pd.Series(prediction_unb, name=legend_unb)
    v2 = pd.Series(prediction_b, name=legend_b)
   
    
    ax = sns.kdeplot(v1, shade=True,color='b')
    sns.kdeplot(v2, shade=True,color='r')

    ax.set(xlabel=label, 
           title=plot_title)

    plt.show()
    
def denormalization(predictions,labels):
    
    """
    This function performs the denormalization step.
    ----------
    prediction: list with the desired property predictions.
    labels: list with the real values of the desired property
    
    Returns
    -------
    predictions: Returns the denormalized predictions.
    """
    for l in range(len(predictions)):
        
        q1 = np.percentile(labels,5)
        q3 = np.percentile(labels,95)
       
        for c in range(len(predictions[0])):
            predictions[l,c] = (q3 - q1) * predictions[l,c] + q1
#            predictions[l,c] = predictions[l,c] * sd_train + m_train
          
    return predictions

def get_reward(predictor, smile):
    """
    This function takes the predictor model and the SMILES string and returns 
    a numerical reward.
    ----------
    predictor: object of the predictive model that accepts a trajectory
        and returns a numerical prediction of desired property for the given 
        trajectory
    smile: SMILES string of the generated molecule
    
    Returns
    -------
    Outputs the reward value for the predicted property of the input SMILES 
    """
    
    list_ss = [smile] 

    pred = predictor.predict(list_ss)
    
    reward = np.exp(-pred/3 + 3)
#    reward = (np.exp(pred/10))**1 - 1
#    reward = (np.exp(pred/12))**1 - 1   reward = np.exp(-(pred-5)/3)
    return reward
  
#    if (pred >= 1) and (pred <= 4):
#        return 0.5
#    else:
#        return -0.01
    
def moving_average(previous_values, new_value, ma_window_size=10): 
    """
    This function performs a simple moving average between the last 9 elements
    and the last one obtained.
    ----------
    previous_values: list with previous values 
    new_value: new value to append, to compute the average with the last ten 
               elements
    
    Returns
    -------
    Outputs the average of the last 10 elements 
    """
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma

def plot_training_progress(training_rewards,training_losses):
    """
    This function plots the progress of the training performance
    ----------
    training_rewards: list with previous reward values
    training_losses: list with previous loss values
    """
    plt.plot(training_rewards)
    plt.xlabel('Training iterations')
    plt.ylabel('Average rewards')
    plt.show()
    plt.plot(training_losses)
    plt.xlabel('Training iterations')
    plt.ylabel('Average losses')
    plt.show()
    
def remove_padding(trajectory):    
    """
    Function that removes the padding character
    Parameters
    ----------
    trajectory: Generated molecule padded with "A"
    Returns
    -------
    SMILE string without the padding character 
    """
    if 'A' in trajectory:
        
        firstA = trajectory.find('A')
        trajectory = trajectory[0:firstA]
    return trajectory
        