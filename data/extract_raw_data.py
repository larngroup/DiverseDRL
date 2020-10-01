# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:42:46 2020

@author: Tiago
"""
import csv

def main():
    
    activity = []
    smiles = []
    mw_values = []
    logP_values = []
        
    with open("data_dna.csv", 'r') as f:
        line = f.readline()
        
        total_molecules = 0
        usefull_molecules = 0
        useless_molecules = 0
        bom = 0
        mau = 0
        
        while line:
            data = line.split(';"')
            
            if total_molecules == 0:
                print("Start data cleaning...\n")
            else:
                
                if len(data[7]) > 6:
                    smi = data[7]
                    act = data[12]
                    mw = data[3]
                    logP = data[5]
                    bom = bom + 1
#                elif data[7] != '""':
#                    smi = data[9]
#                    act = data[13]
#                    mw = data[3]
#                    logP = data[5]
                else:
                    mau  = mau +1
                
                    
                try:
                    act = float(act[0:-1])
                    smiles.append(smi[0:-1])
                    activity.append(act)
                    mw_values.append(mw[0:-1])
                    logP_values.append(logP[0:-1])
                    
                    usefull_molecules += 1
                except:
                    useless_molecules += 1
                    
            line = f.readline()                         
            total_molecules += 1

    filename = 'data_clean_dna.csv'
    with open(filename, mode='w') as w_file:
        file_writer = csv.writer(w_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(['SMILES','pCHEMBL','Molecular Weight','logP'])
        for i,smile in enumerate(smiles):
            file_writer.writerow([smile,activity[i],mw_values[i],logP_values[i]])
    print("File " + filename + " successfully saved!")
if __name__ == '__main__':
    main()