# ECE 239AS Final Project

-- About the project --

## Installation and Setup Instructions

1. After cloning the repository on your local machine, `cd` into it.
2. We use Python3.6 for the project as TensorFlow and Keras are not compatible with Python3.7. Create and activate a virtual environment by running 
`python3.6 -m venv venv` 
`source venv/bin/activate`
3. Install all requirements by running `pip install -r requirements.txt`
4. Make a new folder called `data` and put all of the 6 EEG datasets inside of it.
5. Now from the root folder, launch jupyter notebook in your browser by running `jupyter notebook`


## Notebooks Included:

1. EEG_loading.ipynb: initial CNN and RNN (LSTM, GRU) implementations.
2. Temporal Convolutions.ipynb: reports performance of architecture similar to the Shallow ConvNet from Schirmiesterr et al. 
3. CNN_Trials.ipynb: Contains a series of (mostly unsuccesfull) experiments with Fast Fourier Transforms - including novel ways of reshaping the data, etc.

Note: All code that is called in these notebooks is also included in the zip file.
