import numpy as np

def extract_person_data(X, y, person_train_val, person=0):
    person_X_data = []
    person_y_data = []
    for i in range(person_train_val.shape[0]):
        if person_train_val[i][0] == person:
            person_X_data.append(X[i])
            person_y_data.append(y[i])
    return np.array(person_X_data), np.array(person_y_data)

convert = lambda num : np.array([1.0*(num==769),  1.0*(num==770), 1.0*(num==771), 1.0*(num==772)])

def one_hot_encode(y):
    return np.array([convert(yi) for yi in y])
