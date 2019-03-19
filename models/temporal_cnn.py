import tensorflow as tf
from tensorflow import keras


class TEMPORAL_CNN:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test,dropout=0.0,use_elu=False,use_batchnorm=False):

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.dropout=dropout
        self.use_elu = use_elu
        self.use_batchnorm = use_batchnorm

        # Initialize Keras model
        self.model = keras.Sequential()

    def build_model(self, stride=1, optimizer='adam', input_shape=(1904, 22,1000,1)):

        self.model.add(keras.layers.Conv2D(40, (1,25), strides=stride))
        if self.use_batchnorm: self.model.add(keras.layers.BatchNormalization())
        # self.model.add(keras.layers.MaxPooling2D(pool_size=(1, 1)))
        self.model.add(keras.layers.Activation('elu' if self.use_elu else 'relu'))
        self.model.add(keras.layers.Dropout(self.dropout))

        self.model.add(keras.layers.Conv2D(40, (22,1), strides=stride))
        if self.use_batchnorm: self.model.add(keras.layers.BatchNormalization())
        # self.model.add(keras.layers.MaxPooling2D(pool_size=(1, 1)))
        self.model.add(keras.layers.Activation('elu' if self.use_elu else 'relu'))
        self.model.add(keras.layers.Dropout(self.dropout))

        self.model.add(keras.layers.AveragePooling2D(pool_size=(1, 45)))

        self.model.add(keras.layers.Flatten())

        # self.model.add(keras.layers.Dense(512))
        # if self.use_batchnorm: self.model.add(keras.layers.BatchNormalization())
        # self.model.add(keras.layers.Activation('elu' if self.use_elu else 'relu'))
        # self.model.add(keras.layers.Dropout(self.dropout))

        self.model.add(keras.layers.Dense(4))
        self.model.add(keras.layers.Activation('softmax'))

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        # self.model.build(input_shape=input_shape)

    def evaluate(self):
        return self.model.evaluate(self.X_test, self.y_test, verbose=1)

    def train(self, epochs=5):
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=epochs)

    def show_model(self):
        print(self.model.summary())

    def get_loss_history(self):
        if self.history is None:
            return
        history = self.history
        return history.history['loss'], history.history['val_loss']