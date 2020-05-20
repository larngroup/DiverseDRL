# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 23:22:10 2019

@author: Tiago
"""
import csv
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sascorer_calculator import SAscore
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import time
from bunch import Bunch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import QED
from predictSMILES import *

def load_config(config_file):
    """
    This function loads the configuration file in .json format.
    ----------
    config_file: name of the configuration file
    
    Returns
    -------
    Configuration files
    """
    print("main.py -function- load_config")
    with open(config_file, 'r') as config_file:
        config_dict = json.load(config_file)
        config = Bunch(config_dict)
        exp_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    return config, exp_time;

def reading_csv(config,property_identifier):
    """
    This function loads the labels of the property specified by the identifier.
    ----------
    config: configuration file
    property_identifier: String that identifies the property to optimize.
    
    Returns
    -------
    raw_labels: Returns the labels in a numpy array. 
    """
    if property_identifier == "jak2":
        file_path = config.file_path_jak2
    elif property_identifier == "logP":
        file_path = config.file_path_logP
    elif property_identifier == "kor":
        file_path = config.file_path_kor
        
    raw_labels = []
        
    with open(file_path, 'r') as csvFile:
        reader = csv.reader(csvFile)
        
        it = iter(reader)
#        next(it, None)  # skip first item.    
        for row in it:
            if property_identifier == "jak2" or property_identifier == "kor":
                try:
                    raw_labels.append(float(row[1]))
                except:
                    pass
            elif property_identifier == "logP":
                raw_labels.append(float(row[2]))
    
 
#    labels = []
#    for i in range(len(raw_smiles)):
#        if len(raw_smiles[i])<100:
#            smiles.append(raw_smiles[i])
#            labels.append(raw_labels[i])
            
    return raw_labels

def smilesDict(tokens):
    """
    This function extracts the dictionary that makes the correspondence between
    each charachter and an integer.
    ----------
    tokens: Set of characters that can be on SMILES string
    
    Returns
    -------
    tokenDict: Returns the dictionary that maps characters to integers
    """
    tokenDict = dict((token, i) for i, token in enumerate(tokens))
    return tokenDict

def pad_seq(smiles,tokens,paddSize):
    """
    This function performs the padding of each SMILES string.
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
    for i in range(0,len(smiles)):
        if len(smiles[i]) < maxLength:
            smiles[i] = smiles[i] + tokens[-1]*(maxLength - len(smiles[i]))

    return smiles,maxLength
             
def smiles2idx(smiles,tokenDict):  
    """
    This function transforms each SMILES token to the correspondent integer,
    according the token dictionary.
    ----------
    smiles: Set of SMILES strings with different sizes;
    tokenDict: Dictionary that maps the characters to integers;    
    
    Returns
    -------
    newSmiles: Returns the transformed smiles, with the characters replaced by 
    the respective numbers. 
    """         
    newSmiles =  np.zeros((len(smiles), len(smiles[0])))
    for i in range(0,len(smiles)):
        
        try:
            for j in range(0,len(smiles[i])):
           
                newSmiles[i,j] = tokenDict[smiles[i][j]]
        except:
            pass
    return newSmiles


def scalarization(reward_kor,reward_qed,scalarMode,weights,pred_range):
    """
    This function transforms a vector of two rewards into a unique reward.
    ----------
    reward_kor: Reward obtained from kor property;
    reward_qed: Reward obtained from qed property;
    scalarMode: Type of scalarization;
    weights: List with the weights for two properties: weights[0]: kor, weights[1]: qed
    pred_range: List with the prediction ranges of the reward to normalize the
                obtained values between 0 and 1.
    Returns
    -------
    Returns the scalarized reward according to the scalarization form specified
    """
    
    w_kor = weights[0]
    w_qed = weights[1]
    
    max_kor = pred_range[0]
    min_kor = pred_range[1]
    max_qed = pred_range[2]
    min_qed = pred_range[3]
    
    rescaled_rew_kor = (reward_kor - min_kor)/(max_kor-min_kor)
    
    if rescaled_rew_kor < 0:
        rescaled_rew_kor = 0
    elif rescaled_rew_kor > 1:
        rescaled_rew_kor = 1
        
    rescaled_rew_qed = (reward_qed - min_qed)/(max_qed-min_qed)
    
    if rescaled_rew_qed < 0:
        rescaled_rew_qed = 0
    elif rescaled_rew_qed > 1:
        rescaled_rew_qed = 1
    
    
    if scalarMode == 'linear' or scalarMode == 'ws_linear':
       
        return w_kor*1*rescaled_rew_kor + w_qed*1*rescaled_rew_qed
    
    elif scalarMode == 'chebyshev':
        dist_qed = 0 
        dist_kor = 0
        
        dist_qed = abs(rescaled_rew_qed - max_qed)*w_qed
        dist_kor = abs(rescaled_rew_kor - max_kor)*w_kor
        
        if dist_qed > dist_kor:
            return rescaled_rew_qed
        else:
            return rescaled_rew_kor

def canonical_smiles(smiles,sanitize=True, throw_warning=False):
    """
    Takes list of generated SMILES strings and returns the list of valid SMILES.
    Parameters
    ----------
    smiles: List of SMILES strings to validate
    sanitize: bool (default True)
        parameter specifying whether to sanitize SMILES or not.
            For definition of sanitized SMILES check
            http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol
    throw_warning: bool (default False)
        parameter specifying whether warnings will be thrown if a SMILES is
        invalid
    Returns
    -------
    new_smiles: list of valid SMILES (if it is valid and has <60 characters)
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

def smiles2mol(smiles_list):
    """
    Function that converts a list of SMILES strings to a list of RDKit molecules 
    Parameters
    ----------
    smiles: List of SMILES strings
    ----------
    Returns list of molecules objects 
    """
    mol_list = []
    if isinstance(smiles_list,str):
        mol = Chem.MolFromSmiles(smiles_list, sanitize=True)
        mol_list.append(mol)
    else:
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            mol_list.append(mol)
    return mol_list

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
    Returns
    ----------
    Float indicating the percentage of valid generated molecules
    """
    prediction = np.array(prediction)
    x_label = ''
    plot_title = '' 
    
    print("Proportion of valid SMILES:", valid/n_to_generate)
    
    if property_identifier == "jak2" or property_identifier == "kor":
        print("Max of pIC50: ", np.max(prediction))
        print("Mean of pIC50: ", np.mean(prediction))
        print("Min of pIC50: ", np.min(prediction))
        x_label = "Predicted pIC50"
        plot_title = "Distribution of predicted pIC50 for generated molecules"
    elif property_identifier == "sas":
        print("Max SA score: ", np.max(prediction))
        print("Mean SA score: ", np.mean(prediction))
        print("Min SA score: ", np.min(prediction))
        x_label = "Calculated SA score"
        plot_title = "Distribution of SA score for generated molecules"
    elif property_identifier == "qed":
        print("Max QED: ", np.max(prediction))
        print("Mean QED: ", np.mean(prediction))
        print("Min QED: ", np.min(prediction))
        x_label = "Calculated QED"
        plot_title = "Distribution of QED for generated molecules"  
        
    elif property_identifier == "logP":
        percentage_in_threshold = np.sum((prediction >= 0.0) & 
                                     (prediction <= 5.0))/len(prediction)
        print("Percentage of predictions within drug-like region:", percentage_in_threshold)
        print("Average of log_P: ", np.mean(prediction))
        print("Median of log_P: ", np.median(prediction))
        plt.axvline(x=0.0)
        plt.axvline(x=5.0)
        x_label = "Predicted LogP"
        plot_title = "Distribution of predicted LogP for generated molecules"
        
#    sns.set(font_scale=1)
    ax = sns.kdeplot(prediction, shade=True,color = 'b')
    ax.set(xlabel=x_label,
           title=plot_title)
    plt.show()
    return (valid/n_to_generate)*100
    
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
    Returns
    ----------
    This functions returns the difference between the averages of the predicted
    properties and the % of valid SMILES
    """
    prediction_unb = np.array(prediction_unb)
    prediction_b = np.array(prediction_b)
    
    legend_unb = ''
    legend_b = '' 
    label = ''
    plot_title = ''
    
    print("Proportion of valid SMILES (UNB,B):", valid_unb/n_to_generate,valid_b/n_to_generate )
    if property_identifier == 'jak2' or property_identifier == "kor":
        legend_unb = 'Unbiased pIC50 values'
        legend_b = 'Biased pIC50 values'
        print("Max of pIC50: (UNB,B)", np.max(prediction_unb),np.max(prediction_b))
        print("Mean of pIC50: (UNB,B)", np.mean(prediction_unb),np.mean(prediction_b))
        print("Min of pIC50: (UNB,B)", np.min(prediction_unb),np.min(prediction_b))
    
        label = 'Predicted pIC50'
        plot_title = 'Distribution of predicted pIC50 for generated molecules'
        
    elif property_identifier == "sas":
        legend_unb = 'Unbiased SA score values'
        legend_b = 'Biased SA score values'
        print("Max of SA score: (UNB,B)", np.max(prediction_unb),np.max(prediction_b))
        print("Mean of SA score: (UNB,B)", np.mean(prediction_unb),np.mean(prediction_b))
        print("Min of SA score: (UNB,B)", np.min(prediction_unb),np.min(prediction_b))
    
        label = 'Predicted SA score'
        plot_title = 'Distribution of SA score values for generated molecules'  
    elif property_identifier == "qed":
        legend_unb = 'Unbiased QED values'
        legend_b = 'Biased QED values'
        print("Max of QED: (UNB,B)", np.max(prediction_unb),np.max(prediction_b))
        print("Mean of QED: (UNB,B)", np.mean(prediction_unb),np.mean(prediction_b))
        print("Min of QED: (UNB,B)", np.min(prediction_unb),np.min(prediction_b))
    
        label = 'Predicted QED'
        plot_title = 'Distribution of QED values for generated molecules'  
    elif property_identifier == 'logP':
        legend_unb = 'Unbiased logP values'
        legend_b = 'Biased logP values'
        
        percentage_in_threshold_unb = np.sum((prediction_unb >= 0.0) & 
                                            (prediction_unb <= 5.0))/len(prediction_unb)
        percentage_in_threshold_b = np.sum((prediction_b >= 0.0) & 
                                 (prediction_b <= 5.0))/len(prediction_b)
        print("% of predictions within drug-like region (UNB,B):", 
          percentage_in_threshold_unb,percentage_in_threshold_b)
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
    
    return np.mean(prediction_b) - np.mean(prediction_unb), valid_b/n_to_generate
    
def denormalization(predictions,labels):   
    """
    This function performs the denormalization step.
    ----------
    predictions: list with the desired property predictions.
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

def get_reward(predictor, smile,property_identifier):
    """
    This function takes the predictor model and the SMILES string and returns 
    a numerical reward for the specified property
    ----------
    predictor: object of the predictive model that accepts a trajectory
        and returns a numerical prediction of desired property for the given 
        trajectory
    smile: SMILES string of the generated molecule
    property_identifier: String that indicates the property to optimize
    Returns
    -------
    Outputs the reward value for the predicted property of the input SMILES 
    """
    list_ss = [smile] 

    if property_identifier == 'qed':
        list_ss[0] = Chem.MolFromSmiles(smile)
#        reward_list = SAscore(list_ss)
#        reward = reward_list[0] 
        
        reward = QED.qed(list_ss[0])
        
        reward = np.exp(reward/4)
    else:
    
        pred = predictor.predict(list_ss)
        reward = np.exp(pred/4-1)
#        reward = np.exp(-pred/6+2)

    return reward
  
def padding_one_hot(smiles,tokens): 
    """
    This function performs the padding of one-hot encoding arrays
    ----------
    smiles: Numpy array containing the one-hot encoding vectors
    tokens: List of tokens
    Returns
    -------
    This function outputs an array padded with a padding vector  
    """
    smiles = smiles[0,:,:]
    maxlen = 65
    idx = tokens.index('A')
    padding_vector = np.zeros((1,43))
    padding_vector[0,idx] = 1

#        print('padding smiles...')
    while len(smiles) < maxlen:
        smiles = np.vstack([smiles,padding_vector])
        
    #print("\nSMILES padded:\n", self.smiles,"\n")

    return smiles

def get_reward_MO(predictor, smile):
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
    Outputs the reward values for the KOR and QED properties
    """
    
    list_ss = [smile] 

    mol = smiles2mol(list_ss[0])
    reward_qed = qed_calculator(mol)
    # SAScore prediction
#    list_ss[0] = Chem.MolFromSmiles(smile)
#    sas_list = SAscore(list_ss)
#    sas_smiles = sas_list[0] 
    reward_qed = np.exp(reward_qed[0]/4)

    # pIC50 for kor prediction
    list_ss = [smile] 
    pred = predictor.predict(list_ss)

    reward_kor = np.exp(pred/4 - 1)
#    reward = np.exp(pred/10) - 1
    return reward_kor,reward_qed

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
    
def plot_individual_rewds(rew_qed,rew_kor):
    """
    This function plots the evolution of rewards for each property we want to
    optimize
    ----------
    rew_qed: list with previous reward values for qed property
    rew_kor: list with previous reward values for kor property
    """
    plt.plot(rew_qed)
    plt.xlabel('Training iterations')
    plt.ylabel('Average rewards qed')
    plt.show()
    plt.plot(rew_kor)
    plt.xlabel('Training iterations')
    plt.ylabel('Average losses kor')
    plt.show()
    
def plot_evolution(pred_original,pred_iteration85,property_identifier):
    """
    This function plots the comparison between two distributions of some specified 
    property 
    Parameters
    ----------
    pred_original: list original model predictions
    pred_iteration85: list model predictions after 85 iterations
    """    
    pred_original = np.array(pred_original)
    pred_iteration85 = np.array(pred_iteration85)
 
#    sns.set(font_scale=1)
    legend_0 = "Original Distribution" 
    legend_85 = "Iteration 85" 

    if property_identifier == 'kor':
        label = 'Predicted pIC50 for KOR'
        plot_title = 'Distribution of predicted pIC50 for generated molecules'
    elif property_identifier == 'qed':
        label = 'Predicted QED'
        plot_title = 'Distribution of predicted QED for generated molecules'
#    v1 = pd.Series(pred_original, name=legend_0)
#    v3 = pd.Series(pred_iteration85, name=legend_85)
    v1 = pd.Series(pred_original)
    v3 = pd.Series(pred_iteration85)
 
    ax = sns.kdeplot(v1, shade=True,color='g')
    sns.kdeplot(v3, shade=True,color='r')

    ax.set(xlabel=label, 
           title=plot_title)
    
#    plt.setp(ax.get_legend().get_texts(), fontsize='13') # for legend text
#    plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
#    sns.set(rc={'figure.figsize':(5.0, 3.0)})

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

def generate2file(predictor,generator,configReinforce,n2generate,original_model):
    """
    Function that generates a specified number of SMILES strings and predicts 
    its SA score and possibly its pIC50. This function also saves the valid SMILES
    and respective predictions to a folder called "Generated".

    Parameters
    ----------
    predictor: Predictor model object, to predict the desired property
    generator: Generator model object, to generate new SMILES strings
    configReinforce: Configuration file
    n2generate: Integer specifying the number of SMILES to generate
    original_model: String that indicates which generator was used: The G_0 or 
                    the G_optimized
    Returns
    -------
    This function doesn't return anything, but saves a file with the generated 
    SMILES and the respective predicted properties. 
    """
    if original_model:
        model_type = 'original'
    else:
        model_type = 'biased'
    
    generated = []
    pbar = tqdm(range(n2generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        predictSMILES = PredictSMILES(generator,None,False,0,configReinforce)
        generated.append(predictSMILES.sample())

    sanitized,valid = canonical_smiles(generated,sanitize=True, throw_warning=False)
    unique_smiles = list(np.unique(sanitized))[1:]
    
    mol_list= smiles2mol(unique_smiles)
    prediction_sas = SAscore(mol_list)
    

    if predictor != None:
        prediction_prop = predictor.predict(unique_smiles)
        with open("Generated/smiles_prop_"+model_type+".smi", 'w') as f:
            for i,cl in enumerate(unique_smiles):
                data = str(unique_smiles[i]) + "," +  str(prediction_prop[i]) + "," + str(prediction_sas[i]) 
                f.write("%s\n" % data)    
    else:
        with open("Generated/smiles_sas_"+model_type+".smi", 'w') as f:
            for i,cl in enumerate(unique_smiles):
                data = str(unique_smiles[i]) + "," + str(prediction_sas[i]) 
                f.write("%s\n" % data)  
            
#def countRepeated(smile):
#    
#    n_different = len(list(set(smile)))
#    bonus = 0
#    if n_different < 6:
#        bonus = 0
#    elif n_different >= 6 and n_different <= 11:
#        bonus = 0.3
#    elif n_different > 11:
#        bonus = 0.5
#        
#    
#    repetitions = []
#    repeated = ''
#    for i,symbol in enumerate(smile):
#        if i > 0:
#            if symbol == smile[i-1]:
#                repeated = repeated + symbol
#            else:
#                
#                if len(repeated) > 4:
#                    repetitions.append(repeated)
#
#                repeated = ''
#    penalty = 0
#    
#    for rep in repetitions:
#        penalty = penalty + 0.1*len(rep)
#
#    return penalty,bonus

def compute_thresh(rewards,thresh_set):
    """
    Function that computes the thresholds to choose which Generator will be
    used during the generation step, based on the evolution of the reward values.
    Parameters
    ----------
    rewards: Last 3 reward values obtained from the RL method
    Returns
    -------
    This function returns a threshold depending on the recent evolution of the
    reward. If the reward is increasing the threshold will be lower and vice versa.
    """
    reward_t_2 = rewards[0]
    reward_t_1 = rewards[1]
    reward_t = rewards[2]
    q_t_1 = reward_t_2/reward_t_1
    q_t = reward_t_1/reward_t
    
    if thresh_set == 1:
        thresholds_set = [0.15,0.3,0.2]
    elif thresh_set == 2:
        thresholds_set = [0.05,0.3,0.1] 
    
    threshold = 0
    if q_t_1 < 1 and q_t < 1:
        threshold = thresholds_set[0]
    elif q_t_1 > 1 and q_t > 1:
        threshold = thresholds_set[1]
    else:
        threshold = thresholds_set[2]
        
    return threshold

    
def qed_calculator(mols):
    """
    Function that takes as input a list of SMILES to predict its qed value
    Parameters
    ----------
    mols: list of molecules
    Returns
    -------
    This function returns a list of qed values 
    """
    qed_values = []
    for mol in mols:
        try:
            q = QED.qed(mol)
            qed_values.append(q)
        except: 
            pass
        
    return qed_values

def diversity(smiles_list):
    """
    Function that takes as input a list containing SMILES strings to compute
    its internal diversity
    Parameters
    ----------
    smiles_list: List with valid SMILES strings
    Returns
    -------
    This function returns the internal diversity of the list given as input, 
    based on the computation Tanimoto similarity
    """
    td = 0
    
    fps_A = []
    for i, row in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(row)
            fps_A.append(AllChem.GetMorganFingerprint(mol, 6))
        except:
            print('ERROR: Invalid SMILES!')
            
        
    for ii in range(len(fps_A)):
        for xx in range(len(fps_A)):
            tdi = 1 - DataStructs.TanimotoSimilarity(fps_A[ii], fps_A[xx])
            td += tdi          
      
    td = td/len(fps_A)**2

    return td

def generation(generator,predictor,configReinforce):
    """
    Function that generates a set of molecules to predict their affinity for KOR 
    and QED value.
    Parameters
    ----------
    generator: Object of the generator model
    configReinforce: Configuration file
    Returns
    -------
    This function outputs the minimum and maximum values for both properties. 
    This values will be used to normalize the all assigned rewards between 0 and 1. 
    """    
    generated = []
    pbar = tqdm(range(400))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        predictSMILES = PredictSMILES(generator,None,False,0.1,configReinforce)
        generated.append(predictSMILES.sample())

    sanitized,valid = canonical_smiles(generated,sanitize=True, throw_warning=False)# validar 
    unique_smiles = list(np.unique(sanitized))[1:]
    
    #prediction kor
    prediction_kor = predictor.predict(unique_smiles)
    #prediction sas
    mol_list = smiles2mol(unique_smiles) 
    prediction_qed = qed_calculator(mol_list)
    
    new_pred_qed = [np.exp(element/4) for element in prediction_qed]
    new_pred_kor = [np.exp(element/4 - 1) for element in prediction_kor]
    
    ranges = []
    ranges.append(np.max(new_pred_kor))
    ranges.append(np.min(new_pred_kor))
    ranges.append(np.max(new_pred_qed))
    ranges.append(np.min(new_pred_qed))
    return ranges

def tokenize(smiles,token_table):
    """
    This function transforms a array with padded SMILES strings into an array 
    of lists of tokens
    Parameters
    ----------
    smiles: Array of SMILES strings
    token_table: List with all possible tokens 
    Returns
    -------
    This function returns an array containing lists of tokens to be transformed 
    in numbers in the next step.
    """
    tokenized = []
    
    for smile in smiles:
        N = len(smile)
        i = 0
        j= 0
        token = []
        while (i < N):
            for j in range(len(token_table)):
                symbol = token_table[j]
                if symbol == smile[i:i + len(symbol)]:
                    token.append(symbol)
                    i += len(symbol)
                    break
        diff = N - len(token)
        if diff > 0:
            for ii in range(diff):
                token.append(token_table[-1])
        tokenized.append(token)
    return tokenized

def searchWeights(p,cumulative_rewards,previous_weights,last_3_weights,scalarMode):

    
    keepSearch = True
    if scalarMode == 'linear' or scalarMode == 'chebyshev':
        w_kor = 0.1 * p
    
        if w_kor == 1:
            keepSearch = False
    
    elif scalarMode == 'ws_linear':
        w_best = -1
        if p == 0:
            w_kor = 0
            last_3_weights = []
    
        elif p == 1:
            w_kor = 1
            last_3_weights = []
        elif p == 2:
            w_kor = 0.5
            last_3_weights = [0,1,0.5]
        else:
            if abs(cumulative_rewards[-1] - cumulative_rewards[-2]) < abs(cumulative_rewards[-1] - cumulative_rewards[-3]):
                w_kor = (last_3_weights[-3] + last_3_weights[-1]) / 2
                last_3_weights = [last_3_weights[-3],last_3_weights[-1],w_kor]
            else:
                w_kor = (last_3_weights[-2] + last_3_weights[-1]) / 2
                last_3_weights = [last_3_weights[-1],last_3_weights[-2],w_kor]
    
            if len(cumulative_rewards)>10:
                keepSearch = False
    
                mean_rewards = []
                for i in range(0,len(cumulative_rewards)-2):
                    mean_rewards.append(np.mean([cumulative_rewards[i],cumulative_rewards[i+1]]))
                idx = mean_rewards.index(np.max(mean_rewards))
                w_best = np.mean(previous_weights[idx:idx+1])
    
    
    weights = [w_kor,1-w_kor]
    previous_weights.append(w_kor)
    return weights,previous_weights,last_3_weights,keepSearch


def plot_MO(cumulative_rewards_qed,cumulative_rewards_kor,cumulative_rewards,previous_weights):

    plt.clf()

    # plot the chart
    plt.scatter(cumulative_rewards_kor,cumulative_rewards_qed)

    # zip joins x and y coordinates in pairs
    i = 0
    for x,y in zip(cumulative_rewards_kor,cumulative_rewards_qed):
        
        label = "{:.2f}".format(previous_weights[i]) +", "+ "{:.2f}".format(1-previous_weights[i]) 

        # this method is called for each point
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center
        i+=1
    plt.xlabel("Mean KOR reward")
    plt.ylabel("Mean QED reward")
    #plt.xticks(np.arange(0,10,1))
    #plt.yticks(np.arange(0,5,0.5))

    plt.show()