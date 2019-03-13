import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout

class RNN:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        # Initialize Keras model
        self.model = Sequential()

    def train(self, RNN_architecture=LSTM, activation="sigmoid", \
              optimizer='adam', epochs=5, batch_size=64, dropout=None):
        self.model.add(RNN_architecture(200))

        if dropout != None:
            self.model.add(Dropout(rate=dropout))

        self.model.add(Dense(4, activation=activation))
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=epochs, batch_size=batch_size)

    def evaluate(self):
        return self.model.evaluate(self.X_test, self.y_test, verbose=0)
