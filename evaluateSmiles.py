# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:46:24 2020

@author: Tiago
"""
from tqdm import tqdm
from predictSMILES import *
from utils import canonical_smiles
import matplotlib.pyplot as plt
import numpy as np;

def evaluateSmiles(generator,configReinforce):
    generator.model.load_weights(configReinforce.model_name + ".h5")
    
    file = open("data_10000_len100.smi","r")
    lines = file.readlines()
    
    smiles = []
    for line in lines:
        x = line.split("\t")
        smiles.append(x[0])
        
    
    chembl_data = smiles
    
    generated = []
    pbar = tqdm(range(10000))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        predictSMILES = PredictSMILES(generator, configReinforce)
        generated.append(predictSMILES.sample())

    sanitized,valid = canonical_smiles(generated, sanitize=False, throw_warning=False)# validar 
    unique_smiles = list(np.unique(sanitized))[1:]