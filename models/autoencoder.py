import tensorflow as tf
from tensorflow import keras
import numpy as np

class Autoencoder:
    def __init__(self, X_train, X_val, X_test):

        self.X_train = X_train
        self.y_train = X_train
        self.X_val = X_val
        self.y_val = X_val
        self.X_test = X_test
        self.y_test = X_test

        # Initialize Keras model
        self.model = keras.Sequential()

    def train(self, optimizer='adam', epochs=5, encoded_dim=2200):
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(encoded_dim, activation='relu'))
        self.model.add(keras.layers.Dense(encoded_dim, activation='relu'))
        self.model.add(keras.layers.Dense(22000))
        self.model.add(keras.layers.Reshape((22, 1000)))
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=epochs)

    def evaluate(self):
        return self.model.evaluate(self.X_test, self.y_test, verbose=1)

    def predict(self, X_data):
        return self.model.predict(X_data)
