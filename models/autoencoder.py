import tensorflow as tf
from tensorflow import keras
import numpy as np

class Autoencoder:
    def __init__(self, X_train, X_val, X_test):

        self.X_train = X_train.reshape(X_train.shape[0],np.prod(X_train.shape[1:]))
        self.y_train = X_train.reshape(X_train.shape[0],np.prod(X_train.shape[1:]))
        self.X_val = X_val.reshape(X_val.shape[0],np.prod(X_val.shape[1:]))
        self.y_val = X_val.reshape(X_val.shape[0],np.prod(X_val.shape[1:]))
        self.X_test = X_test.reshape(X_test.shape[0],np.prod(X_test.shape[1:]))
        self.y_test = X_test.reshape(X_test.shape[0],np.prod(X_test.shape[1:]))

        # Initialize Keras model
        self.model = keras.Sequential()

    def train(self, optimizer='adam', epochs=5, encoded_dim=500):
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(encoded_dim, activation='relu'))
        self.model.add(keras.layers.Dense(22000, activation='sigmoid'))

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=epochs)

    def evaluate(self):
        return self.model.evaluate(self.X_test, self.y_test, verbose=1)

    def predict(self, X_data):
        return self.model.predict(X_data)
