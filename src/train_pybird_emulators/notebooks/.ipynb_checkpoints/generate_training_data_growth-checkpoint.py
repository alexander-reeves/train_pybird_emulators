import numpy as np
import utils
import yaml
from scipy.stats import qmc, norm, truncnorm, uniform
import pybird 
import argparse
import pickle
from pybird.correlator import Correlator
from pybird.fftlog import FFTLog
import h5py
import gc

def setup(args):
    '''
    In the set-up we define our latin hypercube grid that we will sample over 
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_samps_per_index", type=int, required=True)
    parser.add_argument("--filename_config", type=str, required=True, help='config with the priors for the different trainable parameters')
    parser.add_argument("--n_gridpoints", type=int, required=True, help='number of gridpoints in the LHC, should be equal to num_jobs*num_samples_per_index')


    args = parser.parse_args(args)


    output_fn = '/cluster/scratch/areeves/pybird_emu_growth_training_data'
   
    num_samps_per_index = args.num_samps_per_index 
    filename_config = args.filename_config
    n_gridpoints = args.n_gridpoints

    config_loaded = utils.read_yaml_file(filename_config)
    parameters_dicts = config_loaded['parameters']

    # Get the number of parameters
    num_params = len(parameters_dicts)

    prior_ranges = np.zeros((num_params, 2))
    for i, parameter in enumerate(parameters_dicts):
        prior_ranges[i] = parameter['prior']

    dimension = len(parameters_dicts)

    lhd = qmc.LatinHypercube(d=dimension).random(n=n_gridpoints)

    return num_samps_per_index, lhd, prior_ranges, output_fn, parameters_dicts


def merge(indices, args):
    
    num_samps_per_index, lhd, prior_ranges, output_fn, parameters_dicts = setup(args)
    
    with h5py.File(output_fn + '/total_data.h5', 'a') as hdf_file:
        # Function to load and save data to HDF5
        def update_or_create_dataset(dataset_name, data):
            if dataset_name in hdf_file:
                # Dataset exists, append the data to the existing dataset
                hdf_file[dataset_name].resize((hdf_file[dataset_name].shape[0] + data.shape[0]), axis=0)
                hdf_file[dataset_name][-data.shape[0]:] = data
            else:
                # Dataset doesn't exist, create it
                if dataset_name == 'params':
                    maxshape = (None, data.shape[1])
                else:
                    maxshape = (None,)
                hdf_file.create_dataset(dataset_name, data=data, maxshape=maxshape)

        for index in indices:
            # Load the processed results for the current index from the npz file
            npz_file_path = output_fn + f'/processed_results_{index}.npz'
            
            # try: 
            with np.load(npz_file_path, mmap_mode='r') as data:
                datasets = ["D", "f", "H", "DA", "params"]
                for dataset in datasets:
                    update_or_create_dataset(dataset, data[dataset])

            # except: 
            #     print(f"could not load file for index: {index}")

def main(indices, args): 

    num_samps_per_index, lhd, prior_ranges, output_fn, parameters_dicts = setup(args)
    #The sampled values from the LHC for each of the input parameters 
    sampled_values = utils.sample_from_hypercube(lhd, prior_ranges) 

    for index in indices: 
        start_index = index * num_samps_per_index
        end_index = start_index + num_samps_per_index
        sub_samples = sampled_values[start_index:end_index]

        D_array = []
        DA_array = []
        f_array = []
        H_array = []
        params_array = []

        for sub_sample_ind in range(sub_samples.shape[0]):
            print("working on sample", start_index+sub_sample_ind)

            params = sub_samples[sub_sample_ind]
            
            D, f, H, DA = utils.get_growth_factor_from_params(params, parameters_dicts)

            #append everything flattened 
            D_array.append(D)
            f_array.append(f)
            H_array.append(H)
            DA_array.append(DA)
            params_array.append(params)

        D_array = np.array(D_array)
        DA_array = np.array(DA_array)
        f_array = np.array(f_array)
        H_array = np.array(H_array)
        params_array = np.array(params_array)

        np.savez(output_fn + f'/processed_results_{index}.npz', params=params_array, D=D_array, DA=DA_array, \
        H=H_array, f=f_array)
        
        yield index
