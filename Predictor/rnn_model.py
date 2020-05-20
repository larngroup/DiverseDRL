# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:17:50 2019

@author: Tiago
"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input,GRU, Dropout
from tensorflow.keras.callbacks import  ModelCheckpoint
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tokens import tokens_table
from tensorflow.keras.callbacks import EarlyStopping
from utils import rmse, r_square, ccc
from matplotlib import pyplot as plt

class BaseModel(object):
    def __init__(self, config, data,drop_out,batch_size,learning_rate,n_units,epochs,activation,
                 rnn,searchParams,descriptor_type):
        self.config = config
        self.model = None
        self.data = data
        self.dropout = drop_out
        self.learning_rate = learning_rate
        self.n_units = n_units
        self.epochs = epochs
        self.activation = activation
        self.batch_size = batch_size
        self.rnn = rnn
        self.searchParams = searchParams
        self.descriptor_type = descriptor_type
        
class Model(BaseModel):
    
    def __init__(self, config, data,drop_out,batch_size,learning_rate,n_units,epochs,activation,
                 rnn,searchParams,descriptor_type):
        super(Model, self).__init__(config, data,drop_out,batch_size,learning_rate,n_units,epochs,
             activation,rnn,searchParams,descriptor_type)
#        self.weight_init = RandomNormal(mean=0.0, stddev=0.05, seed=config.seed)
        token_table = tokens_table()
        self.build_model(token_table.table_len)
        
    
    def build_model(self, n_table):
        self.n_table = n_table
        self.model = Sequential()
        
        X_train = self.data[0]
        y_train = self.data[1]
        X_val = self.data[4]
        y_val = self.data[5]
        
        if self.descriptor_type == 'SMILES':
            self.model = Sequential()
            self.model.add(Input(shape=(self.config.input_length,)))
            self.model.add(Embedding(n_table, self.n_units, input_length=self.config.input_length))
    
            if self.rnn == 'LSTM':
                self.model.add(LSTM(self.n_units, return_sequences=True, input_shape=(None,self.n_units,self.config.input_length),dropout = self.dropout))
                self.model.add(LSTM(self.n_units,dropout = self.dropout))
        
            elif self.rnn == 'GRU':
                self.model.add(GRU(self.n_units, return_sequences=True, input_shape=(None,self.n_units,self.config.input_length),dropout = self.dropout))
                self.model.add(GRU(self.n_units,dropout = self.dropout))
    
            self.model.add(Dense(self.n_units, activation=self.activation))
            #model.add(Dropout(mlp_dropout))
            self.model.add(Dense(1, activation=self.config.activation_dense2))
            
            
        else:
            self.model = Sequential()
            self.model.add(Input(shape=(4096,)))
            self.model.add(Dense(8000, activation='linear'))
            self.model.add(Dropout(0.25))
            self.model.add(Dense(4000, activation='linear'))
            self.model.add(Dropout(0.25))
            self.model.add(Dense(2000, activation='linear'))
            self.model.add(Dropout(0.25))
            self.model.add(Dense(1, activation='linear'))
            
        
        opt = Adam(lr=self.learning_rate, beta_1=self.config.beta_1, beta_2=self.config.beta_2, amsgrad=False)
            	
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7, restore_best_weights=True)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        self.model.compile(loss=self.config.loss_criterium, optimizer = opt, metrics=[r_square,rmse,ccc])

#        lrateh_size=self.config.batch_size,validation_data=(X_test, y_test),callbacks=[lrate])
        result = self.model.fit(X_train, y_train,
          epochs=self.epochs,
          batch_size=self.batch_size,validation_data=(X_val, y_val),callbacks=[es,mc])
        self.model.summary()

        #-----------------------------------------------------------------------------
        # Plot learning curves including R^2 and RMSE
        #-----------------------------------------------------------------------------
        
        # plot training curve for R^2 (beware of scale, starts very low negative)
        plt.plot(result.history['r_square'])
        plt.plot(result.history['val_r_square'])
        plt.title('model R^2')
        plt.ylabel('R^2')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
                   
        # plot training curve for rmse
        plt.plot(result.history['rmse'])
        plt.plot(result.history['val_rmse'])
        plt.title('rmse')
        plt.ylabel('rmse')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        if self.searchParams:
            metrics = self.model.evaluate(self.data[2],self.data[3])        
            print("\n\nMean_squared_error: ",metrics[0],"\nR_square: ", metrics[1], "\nRoot mean square: ",metrics[2], "\nCCC: ",metrics[3])
            values= [self.dropout,self.batch_size,self.learning_rate,self.n_units,
                 self.rnn,self.activation,self.epochs,metrics[0],metrics[1],metrics[2],metrics[3]] 
                          
            file=[i.rstrip().split(',') for i in open('grid_results.csv').readlines()]
            file.append(values)
            file=pd.DataFrame(file)
            file.to_csv('grid_results.csv',header=None,index=None)
        else:
            
            filepath=""+self.config.checkpoint_dir + ""+ self.config.model_name
#             serialize model to JSON
            model_json = self.model.to_json()
            with open(str(filepath + ".json"), "w") as json_file:
                json_file.write(model_json)
                # serialize weights to HDF5
            self.model.save_weights(str(filepath + ".h5"))
            print("Saved model to disk")
        
