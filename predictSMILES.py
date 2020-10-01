import numpy as np
from tqdm import tqdm
from Smiles_to_tokens import SmilesToTokens
import random 

class BasePredictSMILES(object):

    def __init__(self, model_unbiased,model_biased,training, threshold,config):
        self.model_biased = model_biased
        self.model_unbiased = model_unbiased
        self.config = config
        self.training = training
        self.threshold = threshold
        #print("BasePredictSMILES")

class PredictSMILES(BasePredictSMILES):
    """
    Function that samples new SMILES strings using both the unbiased and biased
    Generators depending on the pred-defined threshold
    Parameters
    ----------
    model_unbiased: Unbiased Generator
    model_biased: Optimized Generator
    threshold: Value from which we use the biased Generator and below which we 
    use the initial Generator.
    Returns
    -------
    This function returns the sampled SMILES string.
    """
    def __init__(self, model_unbiased,model_biased,training,threshold,config):
        super(PredictSMILES, self).__init__(model_unbiased,model_biased,training,threshold, config)
        self.model_unbiased = model_unbiased
        self.model_biased = model_biased
        token_table = SmilesToTokens()
        self.table = token_table.table
        self.training = training
        self.threshold = threshold
        
        
    def sample_with_temp(self, preds): 
        """
        Function that selects a token after applying a softmax activation with
        temperature
        Parameters
        ----------
        preds: Probabilities of choosing each character
        temperature: float used to control the randomness of predictions by 
                     scaling the logits before applying softmax
        Returns
        -------
        This function returns a randomly choose character based on all 
        probabilities.
        """
        #streched = np.log(preds) / self.config.sampling_temp
        streched = np.log(preds) /0.92#1.1#0.92#0.91
        streched_probs = np.exp(streched) / np.sum(np.exp(streched))
        return np.random.choice(len(streched), p=streched_probs)

    def sample(self, num=1, minlen=1, maxlen=100, start='G'):
        """
        Function that generates the SMILES string, token by token, depending on 
        the previous computed sequence 
        """        
        sampled = []
        token_table = SmilesToTokens()

        for i in tqdm(range(num)):
            start_a = start      
            sequence = start_a
            contador=0
            while sequence[-1] != 'E' and len(sequence) <= maxlen:
                x, _ = token_table.one_hot_encode(token_table.tokenize(sequence))
                if self.training == True:
                    
                    e = round(random.uniform(0.0, 1.0), 5)
                    
                    if e <self.threshold: # exploring rate
                        preds = self.model_unbiased.model.predict(x)[0][-1]
                    else:
                        preds = self.model_biased.model.predict(x)[0][-1]
                else:
                    preds = self.model_unbiased.model.predict(x)[0][-1]
                    
                next_a = self.sample_with_temp(preds)                
                sequence += self.table[next_a]
                contador=contador + 1
            sequence = sequence[1:].rstrip('E')
            if len(sequence) < minlen:
                continue
            else:
                sampled.append(sequence)
        return sampled