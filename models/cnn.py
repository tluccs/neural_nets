import tensorflow as tf
from tensorflow import keras

class CNN:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        # Initialize Keras model
        self.model = keras.Sequential()

    def train(self, stride=2, optimizer='adam', epochs=5):
        self.model.add(keras.layers.Conv1D(3, 3, strides=stride, padding='same', data_format='channels_last', use_bias=True, input_shape=(22,1000)))
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='same', data_format='channels_last'))
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.Conv1D(3, 3, strides=stride, padding='same', data_format='channels_last', use_bias=True))
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='same', data_format='channels_last'))
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(500))
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.Dense(150))
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.Dense(100))
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.Dense(4))
        self.model.add(keras.layers.Activation('softmax'))

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=epochs)

    def evaluate(self):
        return self.model.evaluate(self.X_test, self.y_test, verbose=1)
