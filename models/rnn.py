import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Permute


class RNN:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, start=0, end=1000, downsample=1):
        self.X_train = X_train[:,:,start:end:downsample]
        self.y_train = y_train
        self.X_val = X_val[:,:,start:end:downsample]
        self.y_val = y_val
        self.X_test = X_test[:,:,start:end:downsample]
        self.y_test = y_test
        # add back some missing data
        """for i in range(1, int(downsample/4)):
            self.X_train = np.concatenate((self.X_train, X_train[:,:,start+i:end:downsample]), axis=0)
            self.X_val = np.concatenate((self.X_val, X_val[:,:,start+i:end:downsample]), axis=0)
            self.y_train = np.concatenate((self.y_train, y_train))
            self.y_val = np.concatenate((self.y_val, y_val))

        print (self.X_train.shape)"""


        # Initialize Keras model
        self.model = Sequential()
        
        # Keep a history of training and val accuracies, and loss
        self.history = None

    def train(self, RNN_architecture=LSTM, activation="sigmoid", \
              optimizer='adam', epochs=5, batch_size=64, dropout=None, units=200, stride=2):
       
        self.model.add(Permute((2,1)))
        self.model.add(RNN_architecture(units))
        self.model.add(Activation('relu'))
        self.model.add(Dense(300, activation='relu'))
        if dropout is not None:
            self.model.add(Dropout(rate=dropout))
        self.model.add(Dense(4, activation=activation))
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        if self.X_val is not None:
            self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), \
                                      epochs=epochs, batch_size=batch_size)
        else:
            self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)
            

    def evaluate(self):
        return self.model.evaluate(self.X_test, self.y_test, verbose=0)
    
    def predict(self, X_data):
        return self.model.predict(X_data)

    def plot(self):
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

    def show_model(self):
        print(self.model.summary())
        
