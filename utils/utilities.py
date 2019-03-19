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

def fft_electrode_data(X, Fs=250, L=4000):
    """
    Runs an FFT on input data

    Parameters
    --------------------
        X     -- numpy array of shape (t,) - data with t timesteps
        Fs    -- the sampling frequency
        L     -- the length of the sample in milliseconds

    Returns
    --------------------
        Px -- the PSD of X - obtained by running an FFT and running some additional processing steps
        f  -- the frequencies that each of the Px corresponds to
    """
    NFFT = Fs//2

    X = np.fft.fft(X, NFFT)

    Px = X * np.conj(X) / (NFFT * L)
    f = Fs/NFFT * np.arange(0, NFFT)

    f = f[f <= Fs//2 + 1]
    Px = Px[0:f.shape[0]]

    return Px, f

def fft_and_reshape(X, Fs=250, L=4000):
    """
    Runs fft on the last axis (assuming that is the timestep axis)

    Parameters
    --------------------
        X     -- numpy array of shape (N, E, T) - data N examples, E electrodes and T timesteps
        Fs    -- the sampling frequency
        L     -- the length of the sample in milliseconds

    Returns
    --------------------
        X_fft -- numpy array of shape (N,C,E) - changing the last axis to C features (C = num_freqs = Fs//4 + 2) and then swapping C & E.
    """
    num_freqs = Fs//4 + 2
    X_fft = np.zeros((X.shape[0], X.shape[1], num_freqs))

    N,E,_ = X.shape

    for i in np.arange(N):
        for j in np.arange(E):
            EEG_trial = X[i][j]
            EEG_trial_FFT,_ = fft_electrode_data(EEG_trial)
            X_fft[i][j] = EEG_trial_FFT

    return X_fft.swapaxes(1,2)

def reshape_fft_1D(X_fft):
    """
    Runs fft on the last axis (assuming that is the timestep axis)

    Parameters
    --------------------
        X_fft -- numpy array of shape (N, C, E) - data N examples, C channels & E electrodes.

    Returns
    --------------------
        X_fft_sp -- numpy array of shape (N, C, H, W) - changing the last axis to P power densities (P = num_freqs = Fs//4 + 2)
    """
    N,C,E= X_fft.shape
    return X_fft.reshape(N,C*E)

def reshape_fft_spatial(X_fft):
    """
    Runs fft on the last axis (assuming that is the timestep axis)

    Parameters
    --------------------
        X_fft -- numpy array of shape (N, C, E) - data N examples, C channels & E electrodes.

    Returns
    --------------------
        X_fft_sp -- numpy array of shape (N, C, H, W) - changing the last axis to P power densities (P = num_freqs = Fs//4 + 2)
    """
    # num_freqs = Fs//4 + 2
    # X_fft = np.zeros((X.shape[0], X.shape[1], num_freqs))

    X_fft = X_fft.swapaxes(2,1) # X_fft is now (N, E, C) again
    N,E,C = X_fft.shape

    H,W = 5,5

    X_fft_sp = np.zeros((N, H, W, C))

    feature_map = {
        '0,2': 0,

        '1,0': 1,
        '1,1': 2,
        '1,2': 3,
        '1,3': 4,
        '1,4': 5,

        '2,0': 7,
        '2,1': 8,
        '2,2': 9,
        '2,3': 10,
        '2,4': 11,

        '3,0': 13,
        '3,1': 14,
        '3,2': 15,
        '3,3': 16,
        '3,4': 17,

        '4,1': 18,
        '4,2': 19,
        '4,3': 20,

        '5,2': 21
    }

    for i in np.arange(N):
        curr_ind  = X_fft[i]
        X_fft_sp_i = np.zeros((H, W, C))

        for h in np.arange(H):
            for w in np.arange(W):
                fmap_idx = "{},{}".format(h,w)
                if fmap_idx in feature_map:
                    X_fft_sp_i[h][w] = curr_ind[feature_map[fmap_idx]]

        X_fft_sp[i] = X_fft_sp_i

    return X_fft_sp
