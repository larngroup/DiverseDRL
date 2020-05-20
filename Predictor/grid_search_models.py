# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 00:57:09 2020

@author: Tiago
"""
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.externals import joblib 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

def grid_search(data,model_type):
    """
    This function performs a grid-search to find the best parameters of each 
    standard QSAR model.
    ----------
    data: Array with training data and respectives labels
    model_type: String that indentifies the model (SVR, RF, and KNN)
    Returns
    -------
    This function returns the best parameters of the experimented model
    """
    
    X_train = data[0]
    y_train = data[1]
    
    if model_type == 'SVM':
       
        model = SVR()
        gs = GridSearchCV(model, {'kernel': ['rbf','linear','poly'], 'C': 2.0 ** np.array([-4, 14]), 'gamma': 2.0 ** np.array([-14, 4])}, n_jobs=5)
        gs.fit(X_train, y_train)
        params = gs.best_params_
        print(params)
        
    elif model_type == 'RF':
        
        model = RandomForestRegressor()
        gs = GridSearchCV(model, {'n_estimators': [500,1000,1500],'max_features': ["auto", "sqrt", "log2"]})
        gs.fit(X_train, y_train)
        params = gs.best_params_
        print(params)
  
    elif model_type == 'KNN':
        
        
        model = KNeighborsRegressor()
        gs = GridSearchCV(model, {'n_neighbors': [3,5,9,11], 'metric': ['euclidean', 'manhattan', 'chebyshev']})
        gs.fit(X_train, y_train)
        params = gs.best_params_
        print(params)
        
       
    return params