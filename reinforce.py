# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 00:39:28 2019

@author: Tiago
"""
from rdkit import Chem
import tensorflow as tf
from prediction import *
import numpy as np
from Smiles_to_tokens import SmilesToTokens
from predictSMILES import *
from model import Model 
from sascorer_calculator import SAscore
from tqdm import tqdm
from utils import * 
from tqdm import trange
from keras.models import Sequential
from keras import losses
import keras.backend as K
from keras import optimizers

sess = tf.compat.v1.InteractiveSession()            
       
class Reinforcement(object):
    def __init__(self, generator, predictor, configReinforce,property_identifier):  
        """
        Constructor for the Reinforcement object.
        Parameters
        ----------
        generator: generative model object that produces string of characters 
            (trajectories)
        predictor: object of any predictive model type
            predictor accepts a trajectory and returns a numerical
            prediction of desired property for the given trajectory
        configReinforce: bunch
            Parameters to use in the predictive model and get_reward function 
        property_identifier: string
            It indicates what property we want to optimize
        Returns
        -------
        object Reinforcement used for implementation of Reinforcement Learning 
        model to bias the Generator
        """

        super(Reinforcement, self).__init__()
        self.generator_unbiased = generator
        self.generator_biased = generator
        self.generator = generator
        self.configReinforce = configReinforce
        self.generator_unbiased.model.load_weights(self.configReinforce.model_name_unbiased)
        self.generator_biased.model.load_weights(self.configReinforce.model_name_unbiased)
        self.token_table = SmilesToTokens()
        self.table = self.token_table.table
        self.predictor = predictor
        self.get_reward = get_reward
        self.property_identifier = property_identifier 
        self.all_rewards = []
        self.all_losses = []
        self.threshold_greedy = 0.1
        self.n_table = len(self.token_table.table)
#        self.sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
#        self.adadelta = optimizers.Adadelta(learning_rate=0.0001, rho=0.95, epsilon=1e-07)
#        self.adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.adam = optimizers.Adam()
    


    def custom_loss(self,aux_matrix):
        """
        Custom loss function with a auxilary matrix, computed for each batch of 
        molecules, to properly perform the multiplication. It ignores the 
        necessary but useless padding vectors 
        """
        def lossfunction(y_true,y_pred):
            return (1/self.configReinforce.batch_size)*K.sum(losses.categorical_crossentropy(y_true,y_pred)*aux_matrix)
       
        return lossfunction

    def get_policy_model(self,aux_array):
        """
        Function that initializes the Genetor, trained with a policy-gradient-based
        method (new optimizer and loss function)
        """
        self.generator_biased = Sequential()
        self.generator_biased = Model(self.configReinforce) 
        self.generator_biased.model.compile(
                optimizer=self.adam,
                loss = self.custom_loss(aux_array))
        
        return self.generator_biased.model

    def policy_gradient(self, gamma=0.97):    
            """
            Implementation of the policy gradient algorithm.
    
            Parameters:
            -----------
            self.n_batch: int
                number of trajectories to sample per batch.    
            gamma: float (default 0.97)
                factor by which rewards will be discounted within one trajectory.
                Usually this number will be somewhat close to 1.0.

            Returns
            -------
            This function returns, at each iteration, the graphs with average 
            reward and averaged loss from the batch of generated trajectories 
            (SMILES).
             """
             
            # Initialize the variable that will contain the output of each prediction
            dimen = len(self.table)
            states = np.empty(0).reshape(0,dimen)
            
            # Re-compile the model to adapt the loss function and optimizer to the RL problem
            self.generator_biased.model = self.get_policy_model(np.arange(43))
            self.generator_biased.model.load_weights(self.configReinforce.model_name_unbiased)
            
            for i in range(self.configReinforce.n_iterations):
                for j in trange(self.configReinforce.n_policy, desc='Policy gradient progress'):
                    
                    cur_reward = 0
                    
                    # Necessary object to transform new generated smiles to one-hot encoding
                    token_table = SmilesToTokens()
                    aux_matrix = np.zeros((65,1))
                    
                    ii = 0
                    for _ in range(self.configReinforce.batch_size):
                    
                        # Sampling new trajectory
                        reward = 0
                       
                        while reward == 0:
                            predictSMILES =  PredictSMILES(self.generator_unbiased,self.generator_biased,True,self.threshold_greedy,self.configReinforce) # generate new trajectory
                            trajectory = predictSMILES.sample() 
           
                            try:                     
                                s = trajectory[0] # because predictSMILES returns a list of smiles strings
                                if 'A' in s: # A is the padding character
                                    s = remove_padding(trajectory[0])
                                    
                                print("Validation of: ", s) 
            
                                mol = Chem.MolFromSmiles(s)
             
                                trajectory = 'G' + Chem.MolToSmiles(mol) + 'E'
#                                trajectory = 'GCCE'
                                
                                reward = self.get_reward(self.predictor,trajectory[1:-1],self.property_identifier)
                                if len(trajectory) > 65:
                                    reward = 0
                                print(reward)
                               
                            except:
                                reward = 0
                                print("\nInvalid SMILES!")
            
                        # Converting string of characters to one-hot enconding
                        trajectory_input,_ = token_table.one_hot_encode(token_table.tokenize(trajectory))
                        ti,_ = token_table.one_hot_encode(token_table.tokenize(trajectory))
                        discounted_reward = reward
                        cur_reward += reward
                 
                        # "Following" the trajectory and accumulating the loss
                        idxs = 0
                        for p in range(1,len(trajectory_input[0,:,])):
                            
                            state = []
                            state = np.reshape(trajectory_input[0,p,:], [1, dimen])
                            idx = np.nonzero(state)
                            state[idx] = state[:,idx[1]]*discounted_reward
#                            output = self.generator_biased.model.predict(trajectory_input[:,0:p,:])
#
                            inp = ti[:,0:p,:]
                  
                            inp_p = padding_one_hot(inp,self.table) # input to padd
                            mat = np.zeros((1,65))
                            mat[:,idxs] = 1

                            if ii == 0:
                                inputs = inp_p
                                aux_matrix = mat
                            else:
                                inputs = np.dstack([inputs,inp_p])
                                aux_matrix = np.dstack([aux_matrix,mat])
            
                            discounted_reward = discounted_reward * gamma
                            
                            states = np.vstack([states, state])
                            ii += 1
                            idxs += 1
                            
                    # Doing backward pass and parameters update
                 
                    states = states[:,np.newaxis,:]
                    inputs = np.moveaxis(inputs,-1,0)

                    aux_matrix = np.squeeze(aux_matrix)
                    aux_matrix = np.moveaxis(aux_matrix,-1,0)
                    
                    self.generator_biased.model.compile(optimizer = self.adam, loss = self.custom_loss(aux_matrix))
                    #update weights based on the provided collection of samples, without regard to any fixed batch size.
                    loss = self.generator_biased.model.train_on_batch(inputs,states) # update the weights with a batch
                    
                    # Clear out variables
                    states = np.empty(0).reshape(0,dimen)
                    inputs = np.empty(0).reshape(0,0,dimen)

                    cur_reward = cur_reward / self.configReinforce.batch_size
               
                    # serialize model to JSON
                    model_json = self.generator_biased.model.to_json()
                    with open(self.configReinforce.model_name_biased + ".json", "w") as json_file:
                        json_file.write(model_json)
                    # serialize weights to HDF5
                    self.generator_biased.model.save_weights(self.configReinforce.model_name_biased + ".h5")
                    print("Updated model saved to disk")
                    
                    if len(self.all_rewards) > 2: # decide the threshold of the next generated batch 
                        self.threshold_greedy = compute_thresh(self.all_rewards[-3:],self.configReinforce.threshold_set)
 
                    self.all_rewards.append(moving_average(self.all_rewards, cur_reward)) 
                    self.all_losses.append(moving_average(self.all_losses, loss))
    
                plot_training_progress(self.all_rewards,self.all_losses)
        
        
    def test_generator(self, n_to_generate,iteration, original_model):
        """
        Function to generate molecules with the specified generator model. 

        Parameters:
        -----------

        n_to_generate: Integer that indicates the number of molecules to 
                    generate
        iteration: Integer that indicates the current iteration. It will be 
                   used to build the filename of the generated molecules                       
        original_model: Boolean that specifies generator model. If it is 
                        'True' we load the original model, otherwise, we 
                        load the fine-tuned model 

        Returns
        -------
        The plot containing the distribuiton of the property we want to 
        optimize. It saves one file containing the generated SMILES strings.
        """
        
        if original_model:
             self.generator.model.load_weights(self.configReinforce.model_name_unbiased)
             print("....................................")
             print("original model load_weights is DONE!")
        else:
             self.generator.model.load_weights(self.configReinforce.model_name_biased + ".h5")
             print("....................................")
             print("updated model load_weights is DONE!")
    
        
        generated = []
        pbar = tqdm(range(n_to_generate))
        for i in pbar:
            pbar.set_description("Generating molecules...")
            predictSMILES = PredictSMILES(self.generator,None,False,self.threshold_greedy,self.configReinforce)
            generated.append(predictSMILES.sample())

        sanitized,valid = canonical_smiles(generated,sanitize=True, throw_warning=False)
        
        san_with_repeated = []
        for smi in sanitized:
            if len(smi) > 1:
                san_with_repeated.append(smi)
        
        unique_smiles = list(np.unique(san_with_repeated))[1:]
        percentage_unq = (len(unique_smiles)/len(san_with_repeated))*100
#        rep = []
#        for smi in unique_smiles:
#            if smi in data_smiles:
#                rep.append(smi)
#        
#        percentage_valid = (valid/len(sanitized))*100
#        percentage_unique = (1 - (len(rep)/len(unique_smiles)))*100        
                
        if self.property_identifier == 'kor':
            prediction = self.predictor.predict(san_with_repeated)
        elif self.property_identifier == 'sas':
            mol_list = smiles2mol(san_with_repeated)
            prediction = SAscore(mol_list)
        elif self.property_identifier == 'qed':
            mol_list = smiles2mol(san_with_repeated)
            prediction = qed_calculator(mol_list)
                                                           
        vld = plot_hist(prediction,n_to_generate,valid,self.property_identifier)
            
        with open(self.configReinforce.file_path_generated + '_' + str(len(san_with_repeated)) + '_iter'+str(iteration)+".smi", 'w') as f:
            for i,cl in enumerate(san_with_repeated):
                data = str(san_with_repeated[i]) + " ," +  str(prediction[i])
                f.write("%s\n" % data)  
                
        # Compute the internal diversity
        div = diversity(unique_smiles)
        
        return generated, prediction,vld,percentage_unq,div
            

    def compare_models(self, n_to_generate,individual_plot):
        """
        Function to generate molecules with the both models

        Parameters:
        -----------
        n_to_generate: Integer that indicates the number of molecules to 
                    generate
                    
        individual_plot: Boolean that indicates if we want to represent the 
                         property distribution of the pre-trained model.

        Returns
        -------
        The plot that contains the distribuitons of the property we want to 
        optimize originated by the original and fine-tuned models. Besides 
        this, it saves a "generated.smi" file containing the valid generated 
        SMILES and the respective property value in "data\" folder
        """

        self.generator.model.load_weights(self.configReinforce.model_name_unbiased)
        print("\n --------- Original model LOADED! ---------")
        
        generated_unb = []
        pbar = tqdm(range(n_to_generate))
        for i in pbar:
            pbar.set_description("Generating molecules...")
            predictSMILES = PredictSMILES(self.generator,None,False,self.threshold_greedy,self.configReinforce)
            generated_unb.append(predictSMILES.sample())
    
        sanitized_unb,valid_unb = canonical_smiles(generated_unb, sanitize=False, throw_warning=False) 
        unique_smiles_unb = list(np.unique(sanitized_unb))[1:]
        
        if self.property_identifier == 'kor':
            prediction_unb = self.predictor.predict(unique_smiles_unb)
        elif self.property_identifier == 'qed': 
            mol_list = smiles2mol(unique_smiles_unb)
            prediction_unb = qed_calculator(mol_list)
        elif self.property_identifier == 'sas':
            mol_list = smiles2mol(unique_smiles_unb)
            prediction_unb = SAscore(unique_smiles_unb)
                 
        if individual_plot:
            plot_hist(prediction_unb,n_to_generate,valid_unb,self.property_identifier)
            
        
        # Load Biased Generator Model 
        self.generator.model.load_weights(self.configReinforce.model_name_biased + ".h5")
        print("\n --------- Updated model LOADED! ---------")
        
        generated_b = []
        pbar = tqdm(range(n_to_generate))
        for i in pbar:
            pbar.set_description("Generating molecules...")
            predictSMILES = PredictSMILES(self.generator,None,False,self.threshold_greedy,self.configReinforce)
            generated_b.append(predictSMILES.sample())
    
        sanitized_b,valid_b = canonical_smiles(generated_b, sanitize=False, throw_warning=False) # validar 
        unique_smiles_b = list(np.unique(sanitized_b))[1:]
        
        if self.property_identifier == 'kor':
            prediction_b = self.predictor.predict(unique_smiles_b)
        elif self.property_identifier == 'qed': 
            mol_list = smiles2mol(unique_smiles_b)
            prediction_b = qed_calculator(mol_list)
        elif self.property_identifier == 'sas':
            mol_list = smiles2mol(unique_smiles_b)
            prediction_b = SAscore(unique_smiles_b)
             
        dif, valid = plot_hist_both(prediction_unb,prediction_b,n_to_generate,valid_unb,valid_b,self.property_identifier)

        div = diversity(unique_smiles_b)
        
        return dif,div,valid
