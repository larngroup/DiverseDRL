# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:08:18 2019

@author: Tiago
"""

class tokens_table(object):
    """
    This class has a list with all the necessary tokens to build the SMILES strings
    Note that a token is not necessarily a character. It can be a two characters like Br.
    ----------
    tokens: List with symbols and characters used in SMILES notation.
    """
    def __init__(self):
        
#        tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
#          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
#          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n', ' ']
        tokens = [
                 'H','Si','Cl', 'Br','B', 'C', 'N', 'O', 'P', 'S', 'F', 'I',
                '(', ')', '[', ']', '=', '#', '@', '*', '%', '0', '1', '2',
                '3', '4', '5', '6', '7', '8', '9', '.', '/', '\\', '+', '-',
                 'c', 'n', 'o', 's','p', ' ']
#        
  
        self.table = tokens
        self.table_len = len(self.table)