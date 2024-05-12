import numpy as np
import yaml
from scipy.stats import qmc, norm, truncnorm, uniform
import pybird 
import argparse
import pickle
from pybird.correlator import Correlator
from pybird.fftlog import FFTLog
import h5py
from train_pybird_emulators.emu_utils import emu_utils
from cosmic_toolbox import logger 
import os
from functools import partial

LOGGER = logger.get_logger(__name__)

def setup(args):
    '''
    In the set-up we define our latin hypercube grid that we will sample over 
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_samps_per_index", type=int, required=True)
    parser.add_argument("--filename_config", type=str, required=True, help='config with the priors for the different trainable parameters')
    parser.add_argument("--n_gridpoints", type=int, required=True, help='number of gridpoints in the LHC, should be equal to num_jobs*num_samples_per_index')
    parser.add_argument(
        "--verbosity",
        default="warning",
        type=str,
        action="store",
        help="Verbosity level",
    )
    parser.add_argument(
        "--scratch",
        type=str,
        required=True,
        help="The scratch directory to store the training data",
    )

    parser.add_argument(
        "--td_type",
        type=str,
        required=True,
        help="The type of training data to compute and store one of ['growth', 'greenfunction', 'pybird_pieces']",
    )

    parser.add_argument(
        "--k_l",
        type=float,
        required=False,
        default=1e-4,
        help="The value of k_l to use for the computation of the pybird pieces training data")
    
    parser.add_argument(
        "--k_r",
        type=float,
        required=False,
        default=0.7,
        help="The value of k_r to use for the computation of the pybird pieces training data")
    
    parser.add_argument(
        "--cov_file",
        type=str,
        required=False,
        default=None,
        help="the covariance matrix file for gaussian sampling")
    
    parser.add_argument(
        "--mu_file",
        type=str,
        required=False,
        default=None,
        help="the mean file for gaussian sampling")

    parser.add_argument("--filename_knots", type=str, default=None, required=False, help='optimal k-knots placement file as a .npy')



    args = parser.parse_args(args)

    td_type = args.td_type
    num_samps_per_index = args.num_samps_per_index 
    filename_config = args.filename_config
    n_gridpoints = args.n_gridpoints
    scratch = args.scratch
    k_l = args.k_l
    k_r = args.k_r
    cov_file = args.cov_file
    mu_file = args.mu_file
    filename_knots = args.filename_knots

    if td_type not in ['growth', 'greenfunction', 'pybird_pieces']:
        raise ValueError(f"td_type must be one of ['growth', 'greenfunction', 'pybird_pieces'] but got {td_type}")
    
    if td_type == "growth":
        LOGGER.info("Computing growth factor training data")
        datasets = ["D", "f", "H", "DA"]
        computation_function = emu_utils.get_growth_factor_from_params
    elif td_type == "greenfunction":
        LOGGER.info("Computing green function training data")
        datasets = ["Y1", "G1t", "V12t", "G1", "f"]
        computation_function = emu_utils.get_green_function_from_params
    elif td_type == "pybird_pieces":
        LOGGER.info("Computing pybird pieces training data")
        datasets = ["P11l", "Pctl", "Ploopl", "IRPs11", "IRPsct", "IRPsloop"]

        LOGGER.info("Setting up pybird")
        N = Correlator()
        #Set up pybird in time unspecified mode for the computation of the pybird pieces training data
        N.set({'output': 'bPk', 'multipole': 3, 'kmax': 0.6,
        'fftaccboost': 2,
        'with_resum': True, 'with_exact_time': True,
        'with_time': False, # time unspecified
        'km': 1., 'kr': 1., 'nd': 3e-4,
        'eft_basis': 'eftoflss', 'with_stoch': True})
        knots = np.load(filename_knots)
        kk = np.logspace(np.log10(k_l), np.log10(k_r), 10000) #ar update make this extremely large such that we are insensitive to interpolation errors!
        computation_function = partial(emu_utils.get_pgg_from_params, N=N,kk=kk, knots=knots)



    # if scratch directory does not exist create it
    os.makedirs(scratch, exist_ok=True)

    config_loaded = emu_utils.read_yaml_file(filename_config)
    parameters_dicts = config_loaded['parameters']

    # Get the number of parameters
    num_params = len(parameters_dicts)

    prior_ranges = np.zeros((num_params, 2))
    for i, parameter in enumerate(parameters_dicts):
        prior_ranges[i] = parameter['prior']

    dimension = len(parameters_dicts)

    lhd = qmc.LatinHypercube(d=dimension).random(n=n_gridpoints)

    # set up logger and print args
    logger.set_logger_level(LOGGER, str(args.verbosity))

    LOGGER.debug(
        "######################## Read in command "
        "line arguments as ############################"
    )
    LOGGER.debug(args)
    LOGGER.debug("####################################################")

    return num_samps_per_index, lhd, prior_ranges, scratch, parameters_dicts, datasets, computation_function, cov_file, mu_file

def main(indices, args): 

    num_samps_per_index, lhd, prior_ranges, scratch, parameters_dicts, datasets, computation_function, cov_file, mu_file = setup(args)
    #The sampled values from the LHC for each of the input parameters 
    if cov_file is not None and mu_file is not None:
        LOGGER.info("Sampling from multivariate gaussian distribution")
        sampled_values = emu_utils.sample_from_hypercube(lhd, prior_ranges, dist="multivariate_gaussian", \
        cov_file = cov_file, mu_file = mu_file)
    
    else:
        LOGGER.info("Sampling from uniform distribution")
        sampled_values = emu_utils.sample_from_hypercube(lhd, prior_ranges) 

    for index in indices: 
        start_index = index * num_samps_per_index
        end_index = start_index + num_samps_per_index
        sub_samples = sampled_values[start_index:end_index]

        output_dict = {}
        for dataset in datasets:
            output_dict[dataset] = []

        params_array = []
        for sub_sample_ind in range(sub_samples.shape[0]):
            LOGGER.info(f"working on sample: {start_index+sub_sample_ind}")

            params = sub_samples[sub_sample_ind]
            outputs = computation_function(params, parameters_dicts)
            for i, dataset in enumerate(datasets):
                output_dict[dataset].append(outputs[i])
            params_array.append(params)

        params_array = np.array(params_array)

        np.savez(scratch + f'/processed_results_{index}.npz', **{dataset: np.array(output_dict[dataset]) for dataset in datasets}, params=params_array)
        
        yield index

    
def merge(indices, args):
    num_samps_per_index, lhd, prior_ranges, scratch, parameters_dicts, datasets, computation_function, cov_file, mu_file = setup(args)
    with h5py.File(scratch + '/total_data.h5', 'a') as hdf_file:
        for index in indices:
            # Load the processed results for the current index from the npz file
            npz_file_path = scratch + f'/processed_results_{index}.npz'
            with np.load(npz_file_path, mmap_mode='r') as data:
                for dataset in datasets+['params']:
                    emu_utils.update_or_create_dataset(dataset, data[dataset], hdf_file)
