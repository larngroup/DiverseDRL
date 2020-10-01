# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 21:36:09 2019

@author: Tiago
"""
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def readFile(filename):
    """
    Function that reads and extracts the SMILES strings from the .smi file
    Parameters
    ----------
    filename: directory of the .smi file
    Returns
    -------
    List of SMILES strings
    """

    file = open(filename,"r")
    lines = file.readlines()
    
    smiles = []
    for line in lines:
        x =line.split("  ")
        smiles.append(x[0].strip())
#    print(len(smiles))
    return smiles

def count_duplicates(seq_a,seq_b): 

    '''takes as argument a sequence and
    returns the number of duplicate elements'''
    seq_both = seq_a + seq_b
#    print(seq_a)
#    print(seq_b)
#    print(seq_both)
    number_duplicates =  len(seq_both) - len(set(seq_both))
    perc_duplicates = (number_duplicates / len(seq_both)) * 100
    return number_duplicates,perc_duplicates

def tanimotoDistance(SmilesSet_A,SmilesSet_B,perc_duplicates):
    """
    Function that plots the distirbution of Tanimoto similarity to the nearest 
    neighbor between two SMILES sets
    Parameters
    ----------
    SmilesSet_A: SMILES set 
    SmilesSet_B: SMILES set 
    Returns
    -------
    This function computes the Tanimoto similarity between each SMILE of set 
    A relatively to each SMILE in set B and chooses the minimum one. Then 
    plots the distribution of nearest neighbor
    """
#    plt.rc('xtick', labelsize=20) 
#    plt.rc('ytick', labelsize=20)
    fp_A = []
    fp_B = []

    for smile_A in SmilesSet_A:
        
        try:
            mol_A = Chem.MolFromSmiles(smile_A)
            fp_A.append(AllChem.GetMorganFingerprint(mol_A,2,useFeatures=True)) 
        except:
            print(smile_A + " - INVALID" + str(len(smile_A)))
        
        
  
    for smile_B in SmilesSet_B:
        try:
        	mol_B = Chem.MolFromSmiles(smile_B)
        	fp_B.append(AllChem.GetMorganFingerprint(mol_B,2,useFeatures=True))
        except:
            print("Invalid: "+ smile_B)

    d = dict()
    h = [];
    for i in range(0,len(fp_A)):
        distance_i = []
        for j in range(0, len(fp_B)):
             dist = DataStructs.TanimotoSimilarity(fp_A[i], fp_B[j])
             distance_i.append(dist)
        
        key = round(max(distance_i),2) # maxporque estamos à procura do nearest neighbor
        h.append(key)
        if key in d:
            d[key] += 1
        else: 
            d[key] = 1
            
    for key in d:    
        d[key] /=  len(fp_A)
    

    lists = sorted(d.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples

#    plt.plot(x, y)
# 
    return h
def plot_hist(h_1,h_2,perc_duplicates_b_o,perc_duplicates_b_u):
    
    plt.ylabel('Relative frequency',fontsize = '15')
    plt.xlabel('Tanimoto similarity to nearest neighbor',fontsize = '15')
    axes = plt.gca()
    axes.set_xlim([0,1])
#    axes.set_ylim([0,1])
    


#    sns.distplot(h, kde=True,
#             hist_kws={'weights': 100*np.full(len(h), 1/len(h))})
   
    
    h_1 = np.array(h_1)
    axes.hist(h_1, alpha=0.5, weights=np.zeros_like(h_1) + 1. / h_1.size)
    axes.text(0.03, 0.25, '% duplicates: '+str(round(perc_duplicates_b_o,2)), style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 4})
    if h_2 != 0:
        h_2 = np.array(h_2)
        axes.hist(h_2,alpha=0.5, weights=np.zeros_like(h_2) + 1. / h_2.size,label='unbiased')
        axes.text(0.03, 0.3, '% duplicates unbiased: '+str(round(perc_duplicates_b_u,2)), style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 4})
    
#    plt.legend(loc='upper right')
    plt.show()
 
            # guardar no dicionário o minimo de distance_i e no valor acrecentar 1 ou criar essa chave se não existir

if __name__ == '__main__':
    """
    This is the main routine that must be called by the command line. The two 
    arguments are the sequences which we want to compute the histogram with 
    the Tanimoto nearest neighbor distance
    """    
#file_a = sys.argv[1]
#file_b = sys.argv[2]  
file_a = "generated_iter22.smi"
#file_b = "data_2000_len60.smi"
file_c = "generated_iter23.smi"
smiles_a = readFile(file_a)
#smiles_b = readFile(file_b)
smiles_c = readFile(file_c)

#n_duplicates_b_o,perc_duplicates_b_o = count_duplicates(smiles_a,smiles_b)
n_duplicates_b_u,perc_duplicates_b_u = count_duplicates(smiles_a,smiles_c)
    
#print("\nNumber of duplicates b-o: %r\nPercentage of duplicates b-o: %f" % (n_duplicates_b_o,
#                                                             perc_duplicates_b_o))
print("\nNumber of duplicates b_u: %r\nPercentage of duplicates b_u: %f" % (n_duplicates_b_u,
                                                             perc_duplicates_b_u))
#h_1 = tanimotoDistance(smiles_a,smiles_b,perc_duplicates_b_o)

h_2 = tanimotoDistance(smiles_a,smiles_c,perc_duplicates_b_u)
#plot_hist(h_1,h_2,perc_duplicates_b_o,perc_duplicates_b_u)
plot_hist(h_2,0,perc_duplicates_b_u,0)