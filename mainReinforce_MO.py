# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 19:39:49 2019

@author: Tiago
"""
from reinforce_MO import Reinforcement
from tqdm import trange
import time
from bunch import Bunch
from keras.models import Sequential
import json
from model import Model 
from matplotlib import pyplot as plt
import numpy as np
from tokens import tokens_table
from prediction_MO import Predictor
from utils import reading_csv

config_file = 'configReinforce.json' 


def get_reward(predictors, smile, configurations):
    """
    Takes the SMILES string and returns a numerical reward.
    ----------
    predictor_logP: object of the predictive model that accepts a trajectory
        and returns a numerical prediction of desired property for the given 
        trajectory
    smile: String
        SMILES string of the generated molecule
    configurations: Bunch 
        Contains the parameters of the gaussian function
    Returns
    -------
    rew: scalar
        Outputs the reward value for the predicted logP value of the input SMILES 
    """
    
    list_ss = [smile] 

    pred_logP = predictors.predict(list_ss,'logP')
    pred_jak2 = predictors.predict(list_ss,'jak2')
   
    pred_jak2 = np.exp(-pred_jak2/4+1)
    
    if (pred_logP >= 1) and (pred_logP <= 4):
        pred_logP = 0.5
    else:
        pred_logP = -0.01
    
    return pred_jak2,pred_logP

    
    
def simple_moving_average(previous_values, new_value, ma_window_size=10):
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma

def load_config(config_file):
    print("main.py -function- load_config")
    with open(config_file, 'r') as config_file:
        config_dict = json.load(config_file)
        config = Bunch(config_dict)
        exp_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    return config, exp_time;

        
def main():
    # load configuration file
    configReinforce,exp_time=load_config(config_file)
    
    # Load generator
    generator_model = Sequential()
    generator_model=Model(configReinforce)
    modelName = configReinforce.model_name
    print("....................................")
    print("model Name is ", modelName)
    generator_model.model.load_weights(modelName)
    print("....................................")
    print("load_weights is DONE!")
    

     # Load the predictor models
    labels_jak2 = reading_csv(configReinforce,"jak2")
    labels_logP = reading_csv(configReinforce,"logP")
    token_table = tokens_table().table
    predictors = Predictor(configReinforce,token_table,labels_jak2,labels_logP)

  
    # Create reinforcement learning object
    RL_logp = Reinforcement(generator_model, predictors, get_reward, configReinforce)
    
#    # estimating with unbiased logP distribution
#    for k in range(4):
#        print("INITIAL GENERATION: " + str(k))
#        smiles_cur, prediction_cur = RL_logp.estimate_and_update(configReinforce.n_to_generate, True)
#    
    rewards = []
    rl_losses = []
    
    for i in range(configReinforce.n_iterations):
        for j in trange(configReinforce.n_policy, desc='Policy gradient...'):
            cur_reward, cur_loss = RL_logp.policy_gradient(i,j)
            rewards.append(simple_moving_average(rewards, cur_reward)) 
            rl_losses.append(simple_moving_average(rl_losses, cur_loss))
    
        plt.plot(rewards)
        plt.xlabel('Training iteration')
        plt.ylabel('Average reward')
        plt.show()
        plt.plot(rl_losses)
        plt.xlabel('Training iteration')
        plt.ylabel('Loss')
        plt.show()
        
 
     for k in range(10):
        print("BIASED GENERATION: " + str(k))
#        smiles_cur, prediction_cur = RL_logp.estimate_and_update(configReinforce.n_to_generate, False)
        RL_logp.compare_models(configReinforce.n_to_generate,True)
if __name__ == '__main__':
    main()