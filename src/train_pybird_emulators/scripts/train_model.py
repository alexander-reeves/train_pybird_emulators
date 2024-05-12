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
from train_pybird_emulators.emu_utils import integrated_model
from train_pybird_emulators.emu_utils import emu_utils
from cosmic_toolbox import logger 
from train_pybird_emulators.emu_utils.k_arrays import k_emu, k_pybird

LOGGER = logger.get_logger(__name__)

def setup(args):
    # The setup function gets executed first before esub starts (useful to create directories for example)

    parser = argparse.ArgumentParser()

    parser.add_argument("--training_data_file", type=str, required=True)
    parser.add_argument("--ntrain", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=False, default=1000)
    parser.add_argument('--log_preprocess', default=False, action='store_true')
    parser.add_argument('--pca_preprocess', default=False, action='store_true')
    parser.add_argument('--mono', default=False, action='store_true')
    parser.add_argument('--quad_hex', default=False, action='store_true')
    parser.add_argument("--rescale", default=False, action='store_true')
    parser.add_argument("--mask_high_k", default=False, action='store_true')
    parser.add_argument("--npca", type=int, required=False, default=64)
    parser.add_argument("--piece_name", type=str, required=False, default=None)
    parser.add_argument("--model_name", type=str, required=False, default=None)
    parser.add_argument("--training_data_file_2", type=str, required=False, default=None)
    parser.add_argument("--ntrain2", type=int, required=False, default=1000)
    parser.add_argument("--layer_size", type=int, required=False, default=512)
    parser.add_argument("--num_layers", type=int, required=False, default=3)
    parser.add_argument("--batch_size", type=int, required=False, default=1024)
    parser.add_argument("--validation_split", type=float, required=False, default=0.2)


    args = parser.parse_args(args)

    if args.rescale:
        cov = emu_utils.get_default_cov()
        flattened_rescale_factor = np.diag(cov)

        if log_preprocess: 
            LOGGER.info("Using log preprocessing so setting rescaling factor to one power of k extra")
            flattened_rescale_factor = np.ones_like(np.diag(cov))/np.repeat(k_emu,3)
    else: 
        flattened_rescale_factor = None

    return args, flattened_rescale_factor 


def main(indices, args):

    for index in indices:

        #check if a GPU is being used by tensorflow 
        LOGGER.info(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
      
        args, flattened_rescale_factor = setup(args)

        x_train, y_train = emu_utils.get_training_data_from_hdf5(args.training_data_file, args.piece_name, args.ntrain, args.mono, args.quad_hex)

        if args.training_data_file_2: 
            x_train_2, y_train_2 = emu_utils.get_training_data_from_hdf5(args.training_data_file_2, args.piece_name, args.ntrain2, args.mono, args.quad_hex)
            x_train = np.concatenate((x_train, x_train_2), axis=0)
            y_train = np.concatenate((y_train, y_train_2), axis=0)

        if flattened_rescale_factor is not None:
            num_patterns = y_train.shape[1] // 97
            rescaling_factor = emu_utils.generate_repeating_array(flattened_rescale_factor, 77, num_patterns//3)  
            if mono: 
                rescaling_factor = emu_utils.generate_repeating_array(flattened_rescale_factor, 77, num_patterns) 
                rescaling_factor = rescaling_factor[:35*77]
            if quadhex:
                rescaling_factor = emu_utils.generate_repeating_array(flattened_rescale_factor, 77, 35) 
                rescaling_factor = rescaling_factor[35*77:]
            if not mono and not quadhex:
                rescaling_factor = rescaling_factor
            rescaling_factor = np.array(rescaling_factor)
        else:
            rescaling_factor = None

        if args.mask_high_k:
            if k_emu.shape[0] != y_train.shape[1]:
                LOGGER.info("starting to mask out some high k")
                num_patterns = y_train.shape[1] // k_pybird.shape[0]
                LOGGER.info(f"num_patterns: {num_patterns}")
                single_mask = np.array([True]*k_emu.shape[0] + [False]*(k_pybird.shape[1] - k_emu.shape[0]))

                # Repeat the mask for num_patterns times
                full_mask = np.tile(single_mask, num_patterns)
                # Apply the mask to remove the high k parts of the array 
                y_train = y_train[:, full_mask]
                y_train, x_train = emu_utils.remove_nan_rows_from_both_arrays(y_train, x_train)
                LOGGER.info("finished masking out some high k")
        
        #Filter out bad indices
        if args.piece_name is not None:
            LOGGER.info(f"filtering out bad indices for piece {piece_name}")
            
            condition_1 = np.any(x_train[:, :-2] > 0, axis=1)
            condition_2 = x_train[:, -1] < 0 
            condition_3 = x_train[:, -2] < 0

            bad_inds = np.where(condition_1 | condition_2 | condition_3)[0]

            if piece_name.startswith("I"):
                print("training IR piece... going to filter out large gradients")
                # Calculate the absolute gradients along each row
                gradients = np.abs(np.diff(y_train, axis=1))

                gradient_threshold = np.quantile(gradients, 0.70)  # top 30% of gradients

                #spikes typically happen around high k 
                spike_positions = np.arange(k_emu.shape[0]-7, gradients.shape[1], k_emu.shape[0])  # Adjust for 0-index and diff output size

                # Condition to identify rows with gradient spikes at specific positions
                condition_4 = np.any(gradients[:, spike_positions] > gradient_threshold, axis=1)

                bad_inds = np.where(condition_1 | condition_2 | condition_3 | condition_4)[0]

            LOGGER.info(f"removing {len(bad_inds)} bad indices")
            x_train = np.delete(x_train, bad_inds, axis=0)
            y_train = np.delete(y_train, bad_inds, axis=0)


        #Are there places where all the columns in the data are zero?
        zero_columns = np.where(np.sum(np.abs(y_train), axis=0) == 0)[0]

        if zero_columns.size>0:
            LOGGER.info(f"removing zero columns for piece {piece_name}")
            #remove and save zero columns indices
            np.save(f"zero_coumns_{piece_name}", zero_columns)
            y_train = np.delete(y_train, zero_columns, axis=1)
            rescaling_factor = np.delete(rescaling_factor, zero_columns, axis=0)

        ##Log pre-processing
        if args.log_preprocess:
            LOGGER.info("using log preprocessing")
            offset = np.amin(y_train, axis=0)
            offset[offset>0] = 0
            y_train = np.log(y_train-2*offset)

        else: 
            offset = False

        ##  PCA pre-processing
        if args.pca_preprocess:
            LOGGER.info("using PCA preprocessing")
            pca_scaler = StandardScaler().fit(y_train)
            pca = PCA(n_components=npca)
            #Fit PCA to standard scaled data 
            normalized_data = pca_scaler.transform(y_train)
            pca.fit(normalized_data)
            y_train = pca.transform(normalized_data)
            rescaling_factor = np.power(pca.explained_variance_, -1) #default for PCA is to use the explained variance to weight the components
            LOGGER.info(f"explained variance: {pca.explained_variance_ratio_}")
            LOGGER.warning("using explained variance to weight the components")
            rescaling_factor = np.array(rescaling_factor)

        else: 
            pca = None
            pca_scaler = None

        input_scaler = StandardScaler().fit(x_train)
        output_scaler = StandardScaler().fit(y_train)

        LOGGER.info(f"x_train shape: {x_train.shape}")
        LOGGER.info(f"y_train shape: {y_train.shape}")

        keras_model = integrated_model.create_model(input_dim=x_train.shape[1], hidden_layers=[256,256,256], output_dim=y_train.shape[1])
        
        #Initialize model and train
        model = integrated_model.IntegratedModel(keras_model, input_scaler=None, output_scaler=None, offset=offset, log_preprocess=args.log_preprocess, temp_file=f"saved_models/{args.model_name}_temp", pca=pca, pca_scaler=pca_scaler, zero_columns=zero_columns, rescaling_factor=rescaling_factor)
        model.train(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=args.validation_split)

        output_path = emu_utils.rel2realpath(f"../data/saved_models/{args.model_name}")
        model.save(output_path)

    yield indices 


