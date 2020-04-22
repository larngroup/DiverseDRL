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
from keras import losses
from tqdm import tqdm
from utils import canonical_smiles,plot_hist,plot_hist_both,scalarization
import matplotlib.pyplot as plt


sess = tf.compat.v1.InteractiveSession()
sess.run(tf.global_variables_initializer()) # initializes weights randomly 
            

def _compute_gradients(tensor, var_list):
      
        grads = tf.gradients(tensor, var_list)
        return [grad if grad is not None else tf.zeros_like(var)
          for var, grad in zip(var_list, grads)]
        
def remove_padding(trajectory):
    if 'A' in trajectory:
        firstA = trajectory.find('A')
        trajectory = trajectory[0:firstA]
    return trajectory
        
class Reinforcement(object):
    def __init__(self, generator, predictors, get_reward, configReinforce):
        
        """
        Constructor for the Reinforcement object.
        Parameters
        ----------
        generator: generative model that produces string of characters 
            (trajectories)
        predictor: object of any predictive model type
            predictor accepts a trajectory and returns a numerical
            prediction of desired property for the given trajectory
        get_reward: function
            custom reward function that accepts a trajectory and predictor 
            and returns a single value of the reward for the given trajectory
        configReinforce: bunch
            Parameters to use in the predictive model and get_reward function      
        Returns
        -------
        object of type Reinforcement used for biasing the properties estimated
        by the predictor of trajectories produced by the generator to maximize
        the custom reward function get_reward.
        """

        super(Reinforcement, self).__init__()
        self.generator = generator
        token_table = SmilesToTokens()
        self.table = token_table.table
        self.configReinforce = configReinforce
        self.predictors = predictors
        self.get_reward = get_reward


    def policy_gradient(self, i,j, n_batch= 7, gamma=0.98):
        
            """
            Implementation of the policy gradient algorithm.
    
            Parameters:
            -----------
    
            i,j: int
                indexes of the number of iterations and number of policies, 
                respectively, to load the models properly, i.e, it's necessary 
                to load the original model just when i=0 and j=0, after that 
                it is loaded the updated one 
            n_batch: int (default 2)
                number of trajectories to sample per batch. When training on GPU
                setting this parameter to to some relatively big numbers can result
                in out of memory error. If you encountered such an error, reduce
                n_batch.
    
            gamma: float (default 0.97)
                factor by which rewards will be discounted within one trajectory.
                Usually this number will be somewhat close to 1.0.

            Returns
            -------
            total_reward: float
                value of the reward averaged through n_batch sampled trajectories
    
            rl_loss: float
                value for the policy_gradient loss averaged through n_batch sampled
                trajectories
    
            """
            
            if i == 0 and j == 0:
                self.generator.model.load_weights(self.configReinforce.model_name)
                print("Original model's weights loaded sucessfully!")
            else:
                self.generator.model.load_weights(self.configReinforce.model_name + ".h5")
                print("Updated model's weights loaded sucessfully!")
            
            
            opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
     
            # loss scalar in tensor format
            loss = tf.zeros(dtype=tf.float32, shape=1) 
        
            total_reward = 0
            
            # Necessary object to transform new generated smiles
            token_table = SmilesToTokens()
            
            
            for _ in range(n_batch):
    
                # Sampling new trajectory
                reward = 0
               
                while reward == 0:
                    predictSMILES =  PredictSMILES(self.generator, self.configReinforce) # generate new trajectory
                    trajectory = predictSMILES.sample() 
                    print(trajectory)
                    try:
                        
                        s = trajectory[0] # because predictSMILES returns a list of smiles strings
                        print(s)
                        if 'A' in s: # A is the padding character
                            s = remove_padding(trajectory[0])
                            
                        print("Validation of: ", s) 

                        mol = Chem.MolFromSmiles(s)
                        
#                        print("Vai smile")
                        trajectory = 'G' + Chem.MolToSmiles(mol) + 'E'
#                        trajectory = 'G' + s + 'E'
#                        print("vem molecule", trajectory)
                        reward_jak2,reward_logP = self.get_reward(self.predictors,trajectory[1:-1],self.configReinforce)
                        
                        print(reward_jak2,reward_logP)
                        reward = scalarization(reward_jak2,reward_logP,'linear')
                        print("\nReward after scalarization: ", reward)            
            
                    except:
                        reward = 0
                        print("zerou porque inválida")
    
    
                # Converting string of characters to one-hot enconding
                trajectory_input,_ = token_table.one_hot_encode(token_table.tokenize(trajectory))
                discounted_reward = reward
                total_reward += reward
         
               
                # "Following" the trajectory and accumulating the loss
                for p in range(1,len(trajectory_input[0,:,])):
                    #print(p)
                    output = self.generator.model.predict(trajectory_input[:,0:p,:])[0][-1]
                    l = losses.categorical_crossentropy(trajectory_input[0,p,:],self.generator.model.output[0,0,:])
                    l = tf.math.multiply(l,tf.constant(discounted_reward,dtype="float32"))
                    loss = tf.math.add(loss,l)
                    
                    discounted_reward = discounted_reward * gamma
                        
                
                # Doing backward pass and parameters update
            loss = tf.math.divide(loss,tf.constant(n_batch,dtype="float32"))
            

            loss_scalar = sess.run(loss,feed_dict={self.generator.model.input: trajectory_input}) 
                        

            # Compute the gradients for a list of variables.
            grads_and_vars = opt.compute_gradients(loss, self.generator.model.trainable_weights)

          
            # Ask the optimizer to apply the calculated gradients.
            sess.run(opt.apply_gradients(grads_and_vars),feed_dict={self.generator.model.input: trajectory_input})

                            
            total_reward = total_reward / n_batch
            
            self.configReinforce.model_name = "generator_model\\LSTM_2layer_adam_d3_updated"
            # serialize model to JSON
            model_json = self.generator.model.to_json()
            with open(self.configReinforce.model_name + ".json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            self.generator.model.save_weights(self.configReinforce.model_name + ".h5")
            print("Updated model saved to disk")
            
            return total_reward, loss_scalar
    
        
    def estimate_and_update(self, n_to_generate, original_model):
        
        if original_model:
             self.generator.model.load_weights(self.configReinforce.model_name)
             print("....................................")
             print("original model load_weights is DONE!")
        else:
             self.generator.model.load_weights(self.configReinforce.model_name + ".h5")
             print("....................................")
             print("updated model load_weights is DONE!")
    
        
        generated = []
        pbar = tqdm(range(n_to_generate))
        for i in pbar:
            pbar.set_description("Generating molecules...")
           # generated.append(generator.evaluate(gen_data, predict_len=120)[1:-1])
            predictSMILES = PredictSMILES(self.generator, self.configReinforce)
            generated.append(predictSMILES.sample())
    
        sanitized,valid = canonical_smiles(generated, sanitize=False, throw_warning=False)# validar 
        unique_smiles = list(np.unique(sanitized))[1:]
 
        prediction_logP = self.predictors.predict(unique_smiles,'logP')
        prediction_jak2 = self.predictors.predict(unique_smiles,'jak2')
                                                           
        plot_hist(prediction_logP, n_to_generate,valid)
        plot_hist(prediction_jak2, n_to_generate,valid)  
        
        return generated
            

    def compare_models(self, n_to_generate,individual_plot):
        
        self.configReinforce.model_name = "generator_model\\LSTM_2layer_adam_d3.hdf5" 
        self.generator.model.load_weights(self.configReinforce.model_name)
        print("\n --------- Original model LOADED! ---------")
        
        generated_unb = []
        pbar = tqdm(range(n_to_generate))
        for i in pbar:
            pbar.set_description("Generating molecules...")
            predictSMILES = PredictSMILES(self.generator, self.configReinforce)
            generated_unb.append(predictSMILES.sample())
    
        sanitized_unb,valid_unb = canonical_smiles(generated_unb, sanitize=False, throw_warning=False) # validar 
        unique_smiles_unb = list(np.unique(sanitized_unb))[1:]
        prediction_unb_jak2 = self.predictors.predict(unique_smiles_unb,'jak2')
        prediction_unb_logP = self.predictors.predict(unique_smiles_unb,'logP')
        
        if individual_plot:
            plot_hist(prediction_unb_jak2,n_to_generate,valid_unb,'jak2')
            plot_hist(prediction_unb_logP,n_to_generate,valid_unb,'logP')
        
        # Load Biased Generator Model
        self.configReinforce.model_name = "generator_model\\LSTM_2layer_adam_d3_updated.h5" 
        self.generator.model.load_weights(self.configReinforce.model_name)
        print("\n --------- Updated model LOADED! ---------")
        
        generated_b = []
        pbar = tqdm(range(n_to_generate))
        for i in pbar:
            pbar.set_description("Generating molecules...")
            predictSMILES = PredictSMILES(self.generator, self.configReinforce)
            generated_b.append(predictSMILES.sample())
    
        sanitized_b,valid_b = canonical_smiles(generated_b, sanitize=False, throw_warning=False) # validar 
        unique_smiles_b = list(np.unique(sanitized_b))[1:]
      
        prediction_b_jak2 = self.predictors.predict(unique_smiles_b,'jak2')
        prediction_b_logP = self.predictors.predict(unique_smiles_b,'logP')

        plot_hist_both(prediction_unb_jak2,prediction_b_jak2,n_to_generate,valid_unb,valid_b,'jak2')
        plot_hist_both(prediction_unb_logP,prediction_b_logP,n_to_generate,valid_unb,valid_b,'logP')        

        with open('generated.smi', 'w') as filehandle:
            filehandle.writelines("%s\n" % smile for smile in generated_b)