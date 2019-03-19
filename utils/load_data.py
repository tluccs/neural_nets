import numpy as np

def load_original_dataset():
    root_dir = "/content/gdrive/My Drive/Colab Notebooks/"
    X_test = np.load(root_dir + "data/X_test.npy")
    y_test = np.load(root_dir+"data/y_test.npy")
    person_train_valid = np.load(root_dir+"data/person_train_valid.npy")
    X_train_valid = np.load(root_dir+"data/X_train_valid.npy")
    y_train_valid = np.load(root_dir+"data/y_train_valid.npy")
    person_test = np.load(root_dir+"data/person_test.npy")

    return X_test, y_test, person_train_valid, X_train_valid, y_train_valid, person_test

def remove_EOG_data(X_train_valid, X_test):
    return X_train_valid[:,0:22,:], X_test[:,0:22,:]

def load_EEG_data():
    X_test, y_test, person_train_valid, X_train_valid, y_train_valid, person_test = load_original_dataset()
    X_train_valid, X_test = remove_EOG_data(X_train_valid, X_test)

    return person_train_valid, X_train_valid, y_train_valid, person_test, X_test, y_test

def split_train_val(X_train_valid, y_train_valid, percent_validation=0.1):
    num_training, _, _ = X_train_valid.shape

    indices = np.arange(num_training)
    np.random.shuffle(indices)

    validation_indices = indices[:int(percent_validation * num_training)]
    training_indices = indices[int(percent_validation * num_training):]

    X_train, y_train = X_train_valid[training_indices,:,:], y_train_valid[training_indices]
    X_val, y_val = X_train_valid[validation_indices,:,:], y_train_valid[validation_indices]

    return X_train, y_train, X_val, y_val


# class EEG_Data(object):
#     def __init__(self, percent_validation=0.1):
#         self.person_test, self.X_test, self.y_test = None, None, None
#         self.person_train_valid, self.X_train_valid, self.y_train_valid = None, None, None

#         self.X_train, self.y_train = None, None
#         self.X_val, self.y_val = None, None

#         self.percent_validation = percent_validation

#     def load_original_dataset(self):
#         self.X_test = np.load("./data/X_test.npy")
#         self.y_test = np.load("./data/y_test.npy")
#         self.person_train_valid = np.load("./data/person_train_valid.npy")
#         self.X_train_valid = np.load("./data/X_train_valid.npy")
#         self.y_train_valid = np.load("./data/y_train_valid.npy")
#         self.person_test = np.load("./data/person_test.npy")

#     def remove_EOG_data(self):
#         self.X_train_valid = self.X_train_valid[:,0:22,:]
#         self.X_test = self.X_test[:,0:22,:]

#     def load(self):
#         self.load_original_dataset()
#         self.remove_EOG_data()
#         return self.person_train_valid, self.X_train_valid, self.y_train_valid, self.person_test, self.X_test, self.y_test

#     def split_train_val(self):
#         num_training, _, _ = self.X_train_valid.shape

#         indices = np.arange(num_training)
#         np.random.shuffle(indices)

#         validation_indices = indices[:int(self.percent_validation * num_training)]
#         training_indices = indices[int(self.percent_validation * num_training):]

#         self.X_train, self.y_train = self.X_train_valid[training_indices,:,:], self.y_train_valid[training_indices]
#         self.X_val, self.y_val = self.X_train_valid[validation_indices,:,:], self.y_train_valid[validation_indices]

#     def shape(self):
#         print ('Training/Valid data shape: {}'.format(self.X_train_valid.shape))
#         print ('Test data shape: {}'.format(self.X_test.shape))
#         print ('Training/Valid target shape: {}'.format(self.y_train_valid.shape))
#         print ('Test target shape: {}'.format(self.y_test.shape))
#         print ('Person train/valid shape: {}'.format(self.person_train_valid.shape))
#         print ('Person test shape: {}'.format(self.person_test.shape))

#         if self.X_train is not None:
#             print ('Training data shape: {}'.format(self.X_train.shape))
#             print ('Training target shape: {}'.format(self.y_train.shape))
#             print ('Validation data shape: {}'.format(self.X_val.shape))
#             print ('Validation target shape: {}'.format(self.y_val.shape))