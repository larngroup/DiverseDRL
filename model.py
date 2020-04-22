from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.initializers import RandomNormal
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Smiles_to_tokens import SmilesToTokens

class BaseModel(object):
    """
    This function constructs the object of the generator model. It defines the 
    model architecture and all the parameters are defined in the configuration 
    file
    ----------
    config: configuration file 
    """
    def __init__(self, config):
        self.config = config
        self.model = None

class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.weight_init = RandomNormal(mean=0.0, stddev=0.05, seed=config.seed)
        token_table=SmilesToTokens()
        self.build_model(len(token_table.table))

    def build_model(self, n_table):
        self.n_table = n_table
        self.model = Sequential()
        self.model.add(
                LSTM(
                    units=self.config.units,
                    input_shape=(None, self.n_table),
                    return_sequences=True,
                    kernel_initializer=self.weight_init,
                    dropout=0.5)#self.config.dropout)
               )
        self.model.add(
                LSTM(
                    units=self.config.units,
                    input_shape=(None, self.n_table),
                    return_sequences=True,
                    kernel_initializer=self.weight_init,
                    dropout=self.config.dropout)
               )
        '''self.model.add(
                LSTM(
                    units=self.config.units,
                    input_shape=(None, self.n_table),
                    return_sequences=True,
                    kernel_initializer=self.weight_init,
                    dropout=self.config.dropout)
               )'''
        self.model.add(
                Dense(
                    units=self.n_table,
                    activation='softmax',
                    kernel_initializer=self.weight_init
                    )
                )
        self.model.compile(
                optimizer=self.config.optimizer,
                loss= 'mse', #'categorical_crossentropy',
                metrics=['accuracy']
                )
