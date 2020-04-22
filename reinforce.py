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
from utils import canonical_smiles,plot_hist,plot_hist_both, remove_padding

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.global_variables_initializer()) # initializes weights randomly 
            
       
class Reinforcement(object):
    def __init__(self, generator, predictor, get_reward, configReinforce,property_identifier):  
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
        self.predictor = predictor
        self.get_reward = get_reward
        self.property_identifier = property_identifier 


    def policy_gradient(self, i,j, n_batch= 2, gamma=0.98):
        
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
#            opt = tf.train.AdamOptimizer(learning_rate=0.0001)
#            sess.run(tf.initialize_all_variables())
            
            if i == 0 and j == 0:
                self.generator.model.load_weights(self.configReinforce.model_name)
                print("Original model's weights loaded sucessfully!")
            else:
                self.generator.model.load_weights(self.configReinforce.model_name + ".h5")
                print("Updated model's weights loaded sucessfully!")
            
            
            opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)

            # loss scalar in tensor format
            self.loss = tf.zeros(dtype=tf.float32, shape=1) 
        
            total_reward = 0
            
            # Necessary object to transform new generated smiles
            token_table = SmilesToTokens()
            
            
            for _ in range(n_batch):
    
                # Sampling new trajectory
                reward = 0
               
                while reward == 0:
                    predictSMILES =  PredictSMILES(self.generator, self.configReinforce) # generate new trajectory
                    trajectory = predictSMILES.sample() 
#                    trajectory = "CNC(=S)Nc1cccc(-c2cnc3ccccc3n2)c1"
                    
                    try:
                        
                        s = trajectory[0] # because predictSMILES returns a list of smiles strings
                        print(s)
                        if 'A' in s: # A is the padding character
                            s = remove_padding(trajectory[0])
                            
                        print("Validation of: ", s) 

                        mol = Chem.MolFromSmiles(s)
     
                        trajectory = 'G' + Chem.MolToSmiles(mol) + 'E'
                        reward = self.get_reward(self.predictor,trajectory[1:-1])
                        
                        print(reward)
                       
                    except:
                        reward = 0
                        print("\nInvalid SMILES!")
    
    
                # Converting string of characters to one-hot enconding
                trajectory_input,_ = token_table.one_hot_encode(token_table.tokenize(trajectory))
                discounted_reward = reward
                total_reward += reward
         
               
                # "Following" the trajectory and accumulating the loss
                for p in range(1,len(trajectory_input[0,:,])):
                    #print(p)
                    output = self.generator.model.predict(trajectory_input[:,0:p,:])[0][-1]
                    c = tf.compat.v1.math.log_softmax(self.generator.model.output[0,0,:])
                    idx = np.nonzero(trajectory_input[0,p,:])
                    l = c[np.asscalar(idx[0])]
#                    l = losses.categorical_crossentropy(-trajectory_input[0,p,:],self.generator.model.output[0,0,:])
                    self.loss = tf.math.subtract(self.loss,tf.math.multiply(l,tf.constant(discounted_reward,dtype="float32")))
#                    self.loss = tf.math.add(self.loss,l)
                    
                    discounted_reward = discounted_reward * gamma
                        
                
                # Doing backward pass and parameters update
            self.loss = tf.math.divide(self.loss,tf.constant(n_batch,dtype="float32"))

            loss_scalar = sess.run(self.loss,feed_dict={self.generator.model.input: trajectory_input}) 
                        

            # Compute the gradients for a list of variables.
            grads_and_vars = opt.compute_gradients(self.loss, self.generator.model.trainable_weights[0:-2])

          
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
    
        
    def test_generator(self, n_to_generate, original_model):
        """
        Function to generate molecules with the specified generator model. 

        Parameters:
        -----------

        n_to_generate: Integer that indicates the number of molecules to 
                    generate
                    
        original_model: Boolean that specifies generator model. If it is 
                        'True' we load the original model, otherwise, we 
                        load the fine-tuned model 

        Returns
        -------
        The plot containing the distribuiton of the property we want to 
        optimize
        """
        
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
            predictSMILES = PredictSMILES(self.generator, self.configReinforce)
            generated.append(predictSMILES.sample())
    
        sanitized,valid = canonical_smiles(generated, sanitize=False, throw_warning=False)# validar 
        unique_smiles = list(np.unique(sanitized))[1:]
 
        prediction = self.predictor.predict(unique_smiles)  
                                                           
        plot_hist(prediction, n_to_generate,valid,self.property_identifier)
            
        return generated, prediction
            

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
        prediction_unb = self.predictor.predict(unique_smiles_unb)  
        
        if individual_plot:
            plot_hist(prediction_unb,n_to_generate,valid_unb,self.property_identifier)
            
        
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
        prediction_b = self.predictor.predict(unique_smiles_b)

        plot_hist_both(prediction_unb,prediction_b,n_to_generate,valid_unb,valid_b,self.property_identifier)
        
#        with open(self.configReinforce.file_path_generated, 'w') as f:
#            f.writelines("%s %s\n" % smile, % for i,smile in enumerate(generated_b)) # FALTA RETIRAR OS []
#        
        with open(self.configReinforce.file_path_generated, 'w') as f:
            for i,cl in enumerate(unique_smiles_b):
                data = str(unique_smiles_b[i]) + "," +  str(prediction_b[i])
                f.write("%s\n" % data)    
            