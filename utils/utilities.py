import numpy as np
import tensorflow as tf

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

def fourier_transform(X_data):
	return np.fft.fft(X_data)

def inverse_fourier(fourier_data):
	return np.fft.ifft(fourier_data)

def crop_fourier(fourier_data, N):
	cropped =  np.copy(fourier_data)
	cropped[:,:,N:cropped.shape[2]-N] = 0
	return cropped

def get_more_trials(X_data, y_data, Ns=[300,350,400]):
	X_train_total = np.copy(X_data)
	y_train_total = np.copy(y_data)
	for N in Ns:
		print("N = " + str(N))
		fourier_X = fourier_transform(X_data)
		cropped = crop_fourier(fourier_X, N)
		X_restored = np.real(np.fft.ifft(cropped_fourier_X_train))
		X_train_total = np.concatenate((X_train_total, X_restored), axis=0)
		y_train_total = np.concatenate((y_train_total, y_data), axis=0)
	return X_train_total, y_train_total

def fft_electrode_data(X):
    L = 4000 # Signal length
    Fs = 250 # sampling freq
    NFFT = Fs//2

    X = np.fft.fft(X, NFFT)

    Px = X * np.conj(X) / (NFFT * L)
    f = Fs/NFFT * np.arange(0, NFFT)

    f = f[f <= Fs//2 + 1]
    Px = Px[0:f.shape[0]]

    return Px, f
