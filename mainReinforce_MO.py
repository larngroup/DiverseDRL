# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 19:39:49 2019

@author: Tiago
"""
from reinforce_MO import Reinforcement
from tqdm import trange
from keras.models import Sequential
from model import Model 
from tokens import tokens_table
from prediction import Predictor
from utils import *

config_file = 'configReinforce.json' 
property_identifier = 'kor'

        
def main():
    # load configuration file
    configReinforce,exp_time=load_config(config_file)
    
    # Load generator
    generator_model = Sequential()
    generator_model=Model(configReinforce)
    generator_model.model.load_weights(configReinforce.model_name_unbiased)

     # Load the predictor models
    predictor = Predictor(configReinforce,property_identifier)

    # Create reinforcement learning object
    RL_obj = Reinforcement(generator_model, predictor, configReinforce)
    
    cumulative_rewards_qed,cumulative_rewards_kor,cumulative_rewards,previous_weights = RL_obj.policy_gradient()
        
#    for k in range(10):
#        print("BIASED GENERATION: " + str(k))
##        smiles_cur, prediction_cur = RL_logp.estimate_and_update(configReinforce.n_to_generate, False)
#        RL_obj.compare_models(configReinforce.n_to_generate,True)
if __name__ == '__main__':
    main()