import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler,LambdaCallback
from sklearn.preprocessing import StandardScaler #for scaling input and output data
from sklearn.preprocessing import RobustScaler #for scaling input and output data
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d, make_interp_spline
import argparse
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model
from tqdm import tqdm
import pickle
from classy import Class
import utils
import integrated_model

def generate_repeating_array(original_array, segment_length, n):
    result_array = []

    for i in range(0, len(original_array), segment_length):
        segment = original_array[i:i + segment_length]
        repeated_segment = np.tile(segment, n)

        result_array.extend(repeated_segment)

    return result_array 


#remove Nans
def remove_nan_rows_from_both_arrays(y_train, x_train):
    """
    Remove rows from data_array that contain any NaN values and remove corresponding rows from x_train.
    Parameters:
    - data_array: 2D numpy array of shape (200000, 10185)
    - x_train: 2D numpy array of shape (200000, 47)
    Returns:
    - Cleaned data_array and x_train with rows containing NaN values removed.
    """
    # Find indices of rows in data_array that have any NaN values
    nan_row_indices = np.where(np.isnan(y_train).any(axis=1))[0]

    # Remove those rows from both arrays
    cleaned_data_array = np.delete(y_train, nan_row_indices, axis=0)
    cleaned_x_train = np.delete(x_train, nan_row_indices, axis=0)

    return cleaned_data_array, cleaned_x_train


def setup(args):
    # The setup function gets executed first before esub starts (useful to create directories for example)

    parser = argparse.ArgumentParser()

    parser.add_argument("--training_data_file", type=str, required=True)
    parser.add_argument("--ntrain", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True, default=1000)
    parser.add_argument('--log_preprocess', default=False, action='store_true')
    parser.add_argument('--pca_preprocess', default=False, action='store_true')
    parser.add_argument('--mono', default=False, action='store_true')
    parser.add_argument('--quad_hex', default=False, action='store_true')
    parser.add_argument("--npca", type=int, required=False, default=64)
    parser.add_argument("--piece_name", type=str, required=True, default="Ploopl")
    parser.add_argument("--model_name", type=str, required=False, default=None)
    parser.add_argument("--training_data_file_2", type=str, required=False, default=None)
    parser.add_argument("--ntrain2", type=int, required=False, default=1000)

    z = 0.5 
    kk = np.logspace(-5, 0, 1000)
    M = Class()
    cosmo = {'omega_b': 0.02235, 'omega_cdm': 0.120, 'h': 0.675, 'ln10^{10}A_s': 3.044, 'n_s': 0.965}
    M.set(cosmo)
    M.set({'output': 'mPk', 'P_k_max_h/Mpc': 1, 'z_max_pk': z})
    M.compute()

    ## [Mpc/h]^3
    Vs = 1.e11      # ideal volume (in [Mpc/h]^3), > 3 x better than DESI / Euclid (BOSS ~ 3e9, DESI ~ 3e10)
    nbar = 5e-3     # ideal nbar (for b1~2) (in [Mpc/h]^3), > 3 x better than DESI / Euclid
    
    pk_lin = np.array([M.pk_lin(k*M.h(), z)*M.h()**3 for k in kk]) # k in Mpc/h, pk in (Mpc/h)^3
    ipk_h = interp1d(kk, pk_lin, kind='cubic')
    kd = np.array([0.001 , 0.0025, 0.005 , 0.0075, 0.01  , 0.0125, 0.015 , 0.0175,
        0.02  , 0.0225, 0.025 , 0.0275, 0.03  , 0.035 , 0.04  , 0.045 ,
        0.05  , 0.055 , 0.06  , 0.065 , 0.07  , 0.075 , 0.08  , 0.085 ,
        0.09  , 0.095 , 0.1   , 0.105 , 0.11  , 0.115 , 0.12  , 0.125 ,
        0.13  , 0.135 , 0.14  , 0.145 , 0.15  , 0.155 , 0.16  , 0.165 ,
        0.17  , 0.175 , 0.18  , 0.185 , 0.19  , 0.195 , 0.2   , 0.205 ,
        0.21  , 0.215 , 0.22  , 0.225 , 0.23  , 0.235 , 0.24  , 0.245 ,
        0.25  , 0.255 , 0.26  , 0.265 , 0.27  , 0.275 , 0.28  , 0.285 ,
        0.29  , 0.295 , 0.3   , 0.31  , 0.32  , 0.33  , 0.34  , 0.35  ,
        0.36  , 0.37  , 0.38  , 0.39  , 0.4])


    args = parser.parse_args(args)
    training_data_file = args.training_data_file
    ntrain = args.ntrain
    epochs = args.epochs
    piece_name = args.piece_name
    log_preprocess = args.log_preprocess
    pca_preprocess = args.pca_preprocess
    model_name=args.model_name
    npca = args.npca
    mono = args.mono
    quad_hex = args.quad_hex
    ntrain2 = args.ntrain2
    training_data_file_2 = args.training_data_file_2

    if mono:
        print("yes mono")


    cov = utils.get_cov(kd, ipk_h, 1., 0, mult=3, nbar=nbar, Vs=Vs)
    flattened_rescale_factor = np.diag(cov)
    # flattened_rescale_factor = np.diag(cov)/np.repeat(kd,3)/np.repeat(kd,3) #add in k^2 scaling

    if log_preprocess: 
        flattened_rescale_factor = np.ones_like(np.diag(cov))/np.repeat(kd,3)

    print("flattened rescaling factor shape", flattened_rescale_factor.shape)

    return training_data_file, model_name, piece_name, ntrain, epochs, log_preprocess, pca_preprocess, npca, flattened_rescale_factor, mono, quad_hex, training_data_file_2, ntrain2


def main(indices, args):

    for index in indices:

        #check if a GPU is being used by tensorflow 
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
      
        training_data_file, model_name, piece_name, ntrain, epochs, log_preprocess, pca_preprocess, npca, flattened_rescale_factor, mono, quadhex, training_data_file_2, ntrain2 = setup(args)

        with h5py.File(training_data_file, 'r') as f:
            print(f.keys())
            print("total number of td points", f['params'].shape)
            x_train = f['params'][:ntrain] #try a test with  just the monopole data

            if mono:
                print("mono selected")
                y_train = f[f'{piece_name}'][:ntrain, :35*97]
            
            if quadhex:
                y_train = f[f'{piece_name}'][:ntrain, 35*97:]

            if not mono and not quadhex:
                print("neither selected")
                y_train = f[f'{piece_name}'][:ntrain]

        if training_data_file_2: 
            with h5py.File(training_data_file_2, 'r') as f:
                x_train_2 = f['params'][:ntrain2] #try a test with  just the monopole data

                if mono:
                    print("mono selected")
                    y_train_2 = f[f'{piece_name}'][:ntrain2, :35*97]
                
                if quadhex:
                    y_train_2 = f[f'{piece_name}'][:ntrain2, 35*97:]

                if not mono and not quadhex:
                    print("neither selected")
                    y_train_2 = f[f'{piece_name}'][:ntrain2]
            
            #concatenate the two training sets
            x_train = np.concatenate((x_train, x_train_2), axis=0)
            y_train = np.concatenate((y_train, y_train_2), axis=0)



        # Determine the number of 97-long patterns within each data vector
        num_patterns = y_train.shape[1] // 97
        print("num_patterns", num_patterns)


        rescaling_factor = generate_repeating_array(flattened_rescale_factor, 77, num_patterns//3)  

        if mono: 
            rescaling_factor = generate_repeating_array(flattened_rescale_factor, 77, num_patterns) 
            rescaling_factor = rescaling_factor[:35*77]

        if quadhex:
            rescaling_factor = generate_repeating_array(flattened_rescale_factor, 77, 35) 
            rescaling_factor = rescaling_factor[35*77:]
        
        if not mono and not quadhex:
            rescaling_factor = rescaling_factor

        rescaling_factor = np.array(rescaling_factor)

        # Create mask for one 97-long array
        single_mask = np.array([True]*77 + [False]*20)

        # Repeat the mask for num_patterns times
        full_mask = np.tile(single_mask, num_patterns)

        # Apply the mask to remove the high k parts of the array 
        y_train = y_train[:, full_mask]

        y_train, x_train = remove_nan_rows_from_both_arrays(y_train, x_train)

        print("ytrain shape", y_train.shape)
        print('rescaling factor shape', rescaling_factor.shape)
        
        condition_1 = np.any(x_train[:, :-2] > 0, axis=1)
        condition_2 = x_train[:, -1] < 0 
        condition_3 = x_train[:, -2] < 0

        bad_inds = np.where(condition_1 | condition_2 | condition_3)[0]

        if piece_name.startswith("I"):
            print("training IR piece... going to filter out large gradients")
            # Calculate the absolute gradients along each row
            gradients = np.abs(np.diff(y_train, axis=1))

            gradient_threshold = np.quantile(gradients, 0.70)  # top 18% of gradients

            #spikes typically happen around high k 
            spike_positions = np.arange(70, gradients.shape[1], 77)  # Adjust for 0-index and diff output size

            # Condition to identify rows with gradient spikes at specific positions
            condition_4 = np.any(gradients[:, spike_positions] > gradient_threshold, axis=1)

            bad_inds = np.where(condition_1 | condition_2 | condition_3 | condition_4)[0]

        print("no. bad inds", bad_inds.shape)
        print(bad_inds)
        x_train = np.delete(x_train, bad_inds, axis=0)
        y_train = np.delete(y_train, bad_inds, axis=0)


        #Are there places where all the columns in the data are zero?
        zero_columns = np.where(np.sum(np.abs(y_train), axis=0) == 0)[0]
        print("zero columns:", zero_columns)

        if zero_columns.size>0:
            #remove and save zero columns indices
            print("removing some columns")
            np.save(f"zero_coumns_{piece_name}", zero_columns)
            y_train = np.delete(y_train, zero_columns, axis=1)
            rescaling_factor = np.delete(rescaling_factor, zero_columns, axis=0)

        ##Log pre-processing
        if log_preprocess:
            #Create the offset vector so we can take a log for pre-processing
            print("taking log of y_train")
            offset = np.amin(y_train, axis=0)
            offset[offset>0] = 0
            print("min", np.amin(y_train-2*offset))
            y_train = np.log(y_train-2*offset)
            # print("using log preprocessing so setting rescaling factor to None")
            # rescaling_factor = np.ones_like(rescaling_factor)

        else: 
            offset = False

        ##  PCA pre-processing
        if pca_preprocess:
            pca_scaler = StandardScaler().fit(y_train)
            pca = PCA(n_components=npca)
            #Fit PCA to standard scaled data 
            normalized_data = pca_scaler.transform(y_train)
            pca.fit(normalized_data)
            y_train = pca.transform(normalized_data)
            print("pca completed")
            rescaling_factor = np.power(pca.explained_variance_, -1) #default for PCA is to use the explained variance to weight the components

            print("using PCA so setting rescaling factor to None")
            rescaling_factor = None

        else: 
            pca = None
            pca_scaler = None

        input_scaler = StandardScaler().fit(x_train)
        output_scaler = StandardScaler().fit(y_train)

        keras_model = integrated_model.create_model(input_dim=x_train.shape[1], hidden_layers=[512,512,512,512], output_dim=y_train.shape[1])

        #Initialize model and train
        model = integrated_model.IntegratedModel(keras_model, input_scaler, output_scaler, offset=offset, log_preprocess=log_preprocess, temp_file=f"saved_models/{model_name}_temp", pca=pca, pca_scaler=pca_scaler, zero_columns=zero_columns, rescaling_factor=rescaling_factor)
        model.train(x_train, y_train, epochs=epochs, batch_size=1024, validation_split=0.2)

        model.save(f"saved_models/80_knots/{model_name}")


    yield indices 


