# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 19:39:49 2019

@author: Tiago
"""
import os
import tensorflow as tf
from keras import backend as K
from reinforce import Reinforcement
from tqdm import trange
import time
from bunch import Bunch
from keras.models import Sequential
import json
from model import Model 
from tokens import tokens_table
from prediction import Predictor
from utils import reading_csv, get_reward, moving_average,plot_training_progress

config_file = 'configReinforce.json' 
property_identifier = 'jak2'

   
from tensorflow.python.framework import ops
ops.get_default_graph()


os.environ["CUDA_VISIBLE_DEVICES"]="0"
session = tf.compat.v1.Session()
K.set_session(session)


from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())


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

        
def main():
    """
    Main routine
    """
    # load configuration file
    configReinforce,exp_time=load_config(config_file)
    
    # Load generator
    generator_model = Sequential()
    generator_model = Model(configReinforce)
    modelName = configReinforce.model_name
    print("....................................")
    print("model Name is ", modelName)
    generator_model.model.load_weights(modelName)
    print("....................................")
    print("load_weights is DONE!")
    

     # Load the predictor model
    labels_raw = reading_csv(configReinforce,property_identifier)
    token_table = tokens_table().table
    predictor = Predictor(configReinforce,token_table,labels_raw,property_identifier)

  
    # Create reinforcement learning object
    RL_logp = Reinforcement(generator_model, predictor, get_reward, configReinforce,property_identifier)
    
#    # estimating with unbiased logP distribution
#    for k in range(4):
#        print("INITIAL GENERATION: " + str(k))
#        smiles_cur, prediction_cur = RL_logp.estimate_and_update(configReinforce.n_to_generate, True)

    training_rewards = []
    training_losses = []
##    
#    for i in range(configReinforce.n_iterations):
#        for j in trange(configReinforce.n_policy, desc='Policy gradient progress'):
#            cur_reward, cur_loss = RL_logp.policy_gradient(i,j)
#            training_rewards.append(moving_average(training_rewards, cur_reward)) 
#            training_losses.append(moving_average(training_losses, cur_loss))
#    
#        
#        plot_training_progress(training_rewards,training_losses)
##        

    for k in range(10):
        print("BIASED GENERATION: " + str(k))
#        smiles_cur, prediction_cur = RL_logp.estimate_and_update(configReinforce.n_to_generate, False)
        RL_logp.compare_models(configReinforce.n_to_generate,True)

if __name__ == '__main__':
    main()