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
import integrated_model

def generate_repeating_array(original_array, segment_length, n):
    result_array = []

    for i in range(0, len(original_array), segment_length):
        segment = original_array[i:i + segment_length]
        repeated_segment = np.tile(segment, n)

        result_array.extend(repeated_segment)

    return result_array 



def setup(args):
    # The setup function gets executed first before esub starts (useful to create directories for example)

    parser = argparse.ArgumentParser()

    parser.add_argument("--training_data_file", type=str, required=True)
    parser.add_argument("--ntrain", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True, default=1000)
    parser.add_argument("--model_name", type=str, required=False, default=None)

    args = parser.parse_args(args)
    training_data_file = args.training_data_file
    ntrain = args.ntrain
    epochs = args.epochs
    model_name=args.model_name


    return training_data_file, model_name, ntrain, epochs


def main(indices, args):

    for index in indices:      
        training_data_file, model_name, ntrain, epochs = setup(args)

        with h5py.File(training_data_file, 'r') as f:
            print(f.keys())
            print("total number of td points", f['params'].shape)
            x_train = f['params'][:ntrain] #try a test with  just the monopole data
            y_train = np.vstack((f['D'][:ntrain],f['f'][:ntrain], f["H"][:ntrain], f["DA"][:ntrain])).T

        print("ytain shape", y_train.shape)


        input_scaler = StandardScaler().fit(x_train)
        output_scaler = StandardScaler().fit(y_train)

        keras_model = integrated_model.create_model(input_dim=x_train.shape[1], hidden_layers=[256, 256, 256], output_dim=y_train.shape[1])

        #Initialize model and train
        model = integrated_model.IntegratedModel(keras_model, input_scaler, output_scaler, temp_file=f"saved_models/{model_name}_temp")
        model.train(x_train, y_train, epochs=epochs, batch_size=1024, validation_split=0.2)

        model.save(f"saved_models/stronger_cuts/{model_name}")


    yield indices 


