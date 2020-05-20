# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:33:59 2020

@author: Tiago
"""
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

def build_models(data,model_type,config,params):
    """
    This function implements the alternative QSAR models - SVR, RF, and KNN
    ----------
    data: List containing Training and Validation data
    model_type: String that specifies the model to be implemented
    config: Configuration files
    params: Set of hyperparameters for the specified model
    Returns
    -------
    This function outputs the trained QSAR model that was specified in the input
    """
     
    X_train = data[0]
    y_train = data[1]
    X_val = data[4]
    y_val = data[5]
    if model_type == 'SVM':
        
        # Fit regression model
        model_to_save = SVR(kernel=params['kernel'], C=params['C'], gamma=params['gamma'])
        
        y_pred = model_to_save.fit(X_train, y_train).predict(X_val)
        
    elif model_type == 'RF':

        # Fit regression model
        model_to_save = RandomForestRegressor(n_estimators=params['n_estimators'], random_state = 0, n_jobs= 1)
        
        y_pred = model_to_save.fit(X_train, y_train).predict(X_val)
    
        
    elif model_type == 'KNN':

        # Fit regression model
        model_to_save = KNeighborsRegressor(n_neighbors=params['n_neighbors'], n_jobs= 1)
        
        y_pred = model_to_save.fit(X_train, y_train).predict(X_val)
 
      
    r2 = r2_score(y_val,y_pred)
    mse = mean_squared_error(y_val,y_pred)
    print("Result set " + ": " + str(r2) + "; "+ str(mse))
    
    # Save the model as a pickle in a file
    filepath=""+config.checkpoint_dir + ""+ config.model_name +".pkl" 
    joblib.dump(model_to_save, filepath)
