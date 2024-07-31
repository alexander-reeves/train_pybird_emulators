import numpy as np
#import utils
from train_pybird_emulators.emu_utils import emu_utils as utils
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
    parser.add_argument("--filename_knots", type=str, required=True, help='optimal k-knots placement file as a .npy')
    parser.add_argument("--n_gridpoints", type=int, required=True, help='number of gridpoints in the LHC, should be equal to num_jobs*num_samples_per_index')


    args = parser.parse_args(args)


    output_fn = '/Users/zhiyulu/Documents/Github/train_pybird_emulators/src/train_pybird_emulators/scripts/pybird_emu_training_data_80_knots'
   
    num_samps_per_index = args.num_samps_per_index 
    filename_config = args.filename_config
    filename_knots = args.filename_knots
    n_gridpoints = args.n_gridpoints

    config_loaded = utils.read_yaml_file(filename_config)
    # print(len(config_loaded))
    # print(config_loaded[0])
    # print(config_loaded[1])
    parameters = config_loaded['parameters']
    knots = np.load(filename_knots)

    # Get the number of parameters
    num_params = len(parameters)


    prior_ranges = np.zeros((num_params, 2))
    for i, parameter in enumerate(parameters):
        print('===========')
        print(parameter)
        prior_ranges[i] = parameter['prior']

    dimension = len(parameters)

    lhd = qmc.LatinHypercube(d=dimension).random(n=n_gridpoints)
    print('.......',lhd.shape)

    N = Correlator()

    #Set up pybird in time unspecified mode
    N.set({'output': 'bPk', 'multipole': 3, 'kmax': 0.6,
       'fftaccboost': 2,
       'with_resum': True, 'with_exact_time': True,
       'with_time': False, # time unspecified
       'km': 1., 'kr': 1., 'nd': 3e-4,
       'eft_basis': 'eftoflss', 'with_stoch': True})


    with open("/Users/zhiyulu/Documents/Science/Cosmology_packages/LSS/FreqCosmo-alex/hpc_work/eft_params_boss_cmass_ngc_l0", 'rb') as f:
        eft_params = pickle.load(f)
    
    # cov_file = f"/Users/zhiyulu/Documents/Science/Cosmology_packages/LSS/FreqCosmo-alex/hpc_work/covariance_{knots.shape[0]}_knots.npy"
    # mu_file = f"/Users/zhiyulu/Documents/Science/Cosmology_packages/LSS/FreqCosmo-alex/hpc_work/mu_{knots.shape[0]}_knots.npy"
    cov_file = f"/Users/zhiyulu/Documents/Science/Cosmology_packages/LSS/FreqCosmo-alex/hpc_work/covariance_55_knots.npy"
    mu_file = f"/Users/zhiyulu/Documents/Science/Cosmology_packages/LSS/FreqCosmo-alex/hpc_work/mu_55_knots.npy"

    return num_samps_per_index, lhd, prior_ranges, output_fn, N, eft_params, knots, cov_file,mu_file


def merge(indices, args):
    print('=================================')
    num_samps_per_index, lhd, prior_ranges, output_fn, N, eft_params, knots, cov_file,mu_file = setup(args)
    
    with h5py.File(output_fn + '/total_data.h5', 'a') as hdf_file:
        # Function to load and save data to HDF5
        def update_or_create_dataset(dataset_name, data):
            if dataset_name in hdf_file:
                # Dataset exists, append the data to the existing dataset
                hdf_file[dataset_name].resize((hdf_file[dataset_name].shape[0] + data.shape[0]), axis=0)
                hdf_file[dataset_name][-data.shape[0]:] = data
            else:
                # Dataset doesn't exist, create it
                hdf_file.create_dataset(dataset_name, data=data, maxshape=(None, None))

        for index in indices:
            # Load the processed results for the current index from the npz file
            npz_file_path = output_fn + f'/processed_results_{index}.npz'
            
            try: 
                with np.load(npz_file_path, mmap_mode='r') as data:
                    datasets = ["P11l", "Pctl", "Ploopl", "IRPs11", "IRPsct", "IRPsloop","params"]
                    # datasets = ["P11l", "Pctl", "Ploopl", "IRPs11", "IRPsct", "params"]

                    for dataset in datasets:
                        update_or_create_dataset(dataset, data[dataset])

            except: 
                print(f"could not load file for index: {index}")

def main(indices, args): 

    num_samps_per_index, lhd, prior_ranges, output_fn, N, eft_params, knots, cov_file,mu_file  = setup(args)

    k_l, k_r = 1e-4, 0.7

    #The sampled values from the LHC for each of the input parameters 
    print('====',lhd.shape)
    sampled_values = utils.sample_from_hypercube(lhd, prior_ranges, dist="multivariate_gaussian", \
    cov_file = cov_file, mu_file = mu_file) 
    print('gt main',sampled_values.shape)
    kk = np.logspace(np.log10(k_l), np.log10(k_r), 10000) #ar update make this extremely large such that we are insensitive to interpolation errors!
    #also match the counter-term with PyBird by default by stopping at k=10^-4/0.7

    for index in indices: 
        start_index = index * num_samps_per_index
        end_index = start_index + num_samps_per_index
        sub_samples = sampled_values[start_index:end_index]

        P11l_array = []
        Pctl_array = []
        Ploopl_array = []
        IRPs11_array = []
        IRPsct_array = []
        IRPsloop_array = []

        params_array = []

        for sub_sample_ind in range(sub_samples.shape[0]):
            print("working on sample", start_index+sub_sample_ind)

            params = sub_samples[sub_sample_ind]

            #logpk_knots, pkmax, f = np.array(params[:-2]), params[-2], params[-1]
            logpk_knots, pkmax, f = np.array(params), params[-2], params[-1]

            print(knots.shape)
            print(logpk_knots.shape)
            ipk_loglog_spline = utils.PiecewiseSpline_jax(knots, logpk_knots)

            pk_lin = np.exp(ipk_loglog_spline(np.log(kk))) #In Mpc/h units everywhere

            print("pk_lin input", pk_lin)

            P11l, Pctl, Ploopl, IRPs11, \
            IRPsct, IRPsloop = utils.get_pgg_from_linps_and_f_and_A(pk_lin, kk, eft_params, N, f, pkmax)

            print("number of Nans p11l", np.count_nonzero(np.isnan(P11l)))
            print("number of Nans Pctl", np.count_nonzero(np.isnan(Pctl)))
            print("number of Nans IRPs11", np.count_nonzero(np.isnan(IRPs11)))
            print("number of Nans IRPsct", np.count_nonzero(np.isnan(IRPsct)))
            print("number of Nans IRPsloop", np.count_nonzero(np.isnan(IRPsloop)))
            print("number of Nans Ploopl", np.count_nonzero(np.isnan(Ploopl)))

            #append everything flattened 
            P11l_array.append(P11l.flatten())
            Pctl_array.append(Pctl.flatten())
            Ploopl_array.append(Ploopl.flatten())
            IRPs11_array.append(IRPs11.flatten())
            IRPsct_array.append(IRPsct.flatten())
            IRPsloop_array.append(IRPsloop.flatten()) 
            params_array.append(params)

        P11l_array = np.array(P11l_array)
        Pctl_array = np.array(Pctl_array)
        Ploopl_array = np.array(Ploopl_array)
        IRPs11_array = np.array(IRPs11_array)
        IRPsct_array = np.array(IRPsct_array)
        IRPsloop_array = np.array(IRPsloop_array)

        print("check this is not crazy")
        print("maximum of array", np.amax(IRPsloop_array))

        params_array = np.array(params_array)

        np.savez(output_fn + f'/processed_results_{index}.npz', params=params_array, P11l=P11l_array, \
        Pctl=Pctl_array, Ploopl=Ploopl_array, IRPs11=IRPs11_array, IRPsct=IRPsct_array, IRPsloop=IRPsloop_array)
        
        yield index
