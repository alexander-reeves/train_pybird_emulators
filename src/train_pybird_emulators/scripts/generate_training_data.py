import os 
os.environ["JAX_PLATFORMS"] = "cpu"
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
import jax.numpy as jnp
import jax 
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from scipy.interpolate import interp1d
from scipy.stats import norm
from tqdm import tqdm
from pybird.correlator import Correlator
LOGGER = logger.get_logger(__name__)


def setup(args):
    """
    In the set-up we define our latin hypercube grid that we will sample over
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_samps_per_index", type=int, required=True)
    parser.add_argument(
        "--filename_config",
        type=str,
        required=True,
        help="config with the priors for the different trainable parameters",
    )
    parser.add_argument(
        "--n_gridpoints",
        type=int,
        required=True,
        help="number of gridpoints in the LHC, should be equal to num_jobs*num_samples_per_index",
    )
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
        help="The value of k_l to use for the computation of the pybird pieces training data",
    )

    parser.add_argument(
        "--k_r",
        type=float,
        required=False,
        default=0.7,
        help="The value of k_r to use for the computation of the pybird pieces training data",
    )

    parser.add_argument(
        "--nf_model_file",
        type=str,
        required=False,
        default=None,
        help="a normalizing flwo model for matched sampling",
    )

    parser.add_argument(
        "--gc_dir",
        type=str,
        required=False,
        default=None,
        help="the directory containing the gaussian copula information",
    )

    parser.add_argument(
        "--cov_file",
        type=str,
        required=False,
        default=None,
        help="the covariance matrix file for gaussian sampling",
    )

    parser.add_argument(
        "--mu_file",
        type=str,
        required=False,
        default=None,
        help="the mean file for gaussian sampling",
    )

    parser.add_argument(
        "--filename_knots",
        type=str,
        default=None,
        required=False,
        help="optimal k-knots placement file as a .npy",
    )

    parser.add_argument(
        "--emu_inputs_file",
        type=str,
        default=None,
        required=False,
        help="file containing the emu inputs",
    )

    parser.add_argument(
        "--cov_factor",
        type=float,
        default=1.0,
        required=False,
        help="factor to multiply the covariance matrix by",

    )

    parser.add_argument(
        "--nf_factor",
        type=float,
        default=1.0,
        required=False,
        help="factor to multiply the sampled base sigma when using the nf sampling",

    )

    parser.add_argument(
        "--gc_factor",
        type=float,
        default=1.0,
        required=False,
        help="factor to multiply the sampled base sigma when using the gaussian copula sampling",)

    parser.add_argument(
        "--clip_distributions",
        action="store_true",
        help="clip the distributions to take samples only up to 0",
    )

    # add seed for randomizing when making new datasets `
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42
    )

    # add an index offset so that we dont have to go over the mem limit to generate new samples `
    parser.add_argument(
        "--ind_offset",
        type=int,
        required=False,
        default=0
    )

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
    cov_factor = args.cov_factor
    clip_distributions = args.clip_distributions
    ind_offset = args.ind_offset
    nf_factor = args.nf_factor
    nf_model_file = args.nf_model_file
    gc_dir = args.gc_dir
    gc_factor = args.gc_factor
    emu_inputs_file = args.emu_inputs_file


    if td_type not in ["growth_and_green", "full_bpk", "pybird_pieces"]:
        raise ValueError(
            f"td_type must be one of ['growth_and_green', 'full_bpk' or  'pybird_pieces'] but got {td_type}"
        )

    if td_type == "growth_and_green":
        LOGGER.info("Computing growth and green function training data")
        datasets = ["D", "f", "H", "DA","Y1", "G1t", "V12t", "G1", "fplus"]
        computation_function = emu_utils.get_growth_and_greens_from_params

    elif td_type == "full_bpk":
        LOGGER.info("Computing full bpk training data")
        datasets = ["bpk", "kk", "pk_lin", "f1", "D1"]
        computation_function = emu_utils.get_bpk_full_from_params
    
    elif td_type == "pybird_pieces":
        LOGGER.info("Computing pybird pieces training data")
        datasets = ["P11l", "Pctl", "Ploopl", "IRPs11", "IRPsct", "IRPsloop"]

        LOGGER.info("Setting up pybird")
        N = Correlator()
        # Set up pybird in time unspecified mode for the computation of the pybird pieces training data
        N.set(
            {
                "output": "bPk",
                "multipole": 3,
                "kmax": 0.4, #new kmax so dont have to filter out the large ks! 
                "fftaccboost": 2,
                "with_resum": True,
                "with_exact_time": True,
                "with_time": False,  # time unspecified
                "km": 1.0,
                "kr": 1.0,
                "nd": 3e-4,
                "eft_basis": "eftoflss",
                "with_stoch": True,
            }
        )
        knots = np.load(filename_knots)
        kk = np.logspace(
            np.log10(k_l), np.log10(k_r), 1000
        )  # ar update make this large number of k-points (matching bank) such that we are insensitive to interpolation errors!
        computation_function = partial(
            emu_utils.get_pgg_from_params, N=N, kk=kk, knots=knots
        )

    # if scratch directory does not exist create it
    os.makedirs(scratch, exist_ok=True)

    config_loaded = emu_utils.read_yaml_file(filename_config)
    parameters_dicts = config_loaded["parameters"]

    num_params = len(parameters_dicts)

    prior_ranges = np.zeros((num_params, 2))
    for i, parameter in enumerate(parameters_dicts):
        prior_ranges[i] = parameter["prior"]

    # set up logger and print args
    logger.set_logger_level(LOGGER, str(args.verbosity))

    LOGGER.debug(
        "######################## Read in command "
        "line arguments as ############################"
    )
    LOGGER.debug(args)
    LOGGER.debug("####################################################")

    return (
        num_samps_per_index,
        prior_ranges,
        scratch,
        parameters_dicts,
        datasets,
        n_gridpoints,
        computation_function,
        cov_file,
        mu_file,
        cov_factor,
        clip_distributions,
        ind_offset,
        nf_model_file,
        nf_factor,
        gc_dir,
        gc_factor,
        emu_inputs_file,
    )


def main(indices, args):

    (
        num_samps_per_index,
        prior_ranges,
        scratch,
        parameters_dicts,
        datasets,
        n_gridpoints,
        computation_function,
        cov_file,
        mu_file,
        cov_factor,
        clip_distributions,
        ind_offset,
        nf_model_file,
        nf_factor,
        gc_dir,
        gc_factor,
        emu_inputs_file,
    ) = setup(args)
    # The sampled values from the LHC for each of the input parameters

    if nf_model_file is not None:
        n_feature = 82
        n_layers = 12
        n_hidden=3
        n_hiddens = [64]*3
        n_bins = 10
        key, subkey = jax.random.split(jax.random.PRNGKey(1))
        model = MaskedCouplingRQSpline(
            n_feature,
            n_layers,
            n_hiddens,
            n_bins,
            subkey,
            data_cov=None,
            data_mean=None,
        )
        model = model.load_model(nf_model_file)
    
    elif gc_dir is not None:
        LOGGER.info("Sampling from gaussian copula distribution")
        LOGGER.info("Applying a covariance factor of {}".format(gc_factor))
        with open(gc_dir + '/empirical_gc_cdf_data.pkl', 'rb') as f:
            empirical_cdf_data = pickle.load(f)
        # Reconstruct empirical CDF functions
        empirical_cdfs = []

        for cdf_data in empirical_cdf_data:
            cdf_func = interp1d(
                cdf_data['x'],
                cdf_data['y'],
                bounds_error=False,
                fill_value=(0.0, 1.0)
            )
            empirical_cdfs.append(cdf_func)
        
        loaded_params = np.load(gc_dir +'/gaussian_copula_params.npz')
        mean_gaussian = loaded_params['mean_gaussian']
        cov_gaussian = loaded_params['cov_gaussian']
    
        def generate_new_samples(empirical_cdfs, mean_gaussian, cov_gaussian, num_new_samples, rng_key):
            L = jnp.linalg.cholesky(cov_gaussian*gc_factor)
            z = jax.random.normal(rng_key, (num_new_samples, mean_gaussian.shape[0]))
            gaussian_samples = mean_gaussian + z @ L.T
            gaussian_samples = np.nan_to_num(gaussian_samples, nan=0.0, posinf=5.0, neginf=-5.0)
            uniform_samples = norm.cdf(gaussian_samples)
            epsilon = 1e-6
            uniform_samples = np.clip(uniform_samples, epsilon, 1 - epsilon)
            generated_samples = np.zeros_like(uniform_samples)
            for i in range(uniform_samples.shape[1]):
                data_points = empirical_cdfs[i].x
                cdf_values = empirical_cdfs[i].y
                # Create inverse CDF function
                inverse_cdf = interp1d(
                    cdf_values,
                    data_points,
                    bounds_error=False,
                    fill_value=(data_points[0], data_points[-1])
                )
                generated_samples[:, i] = inverse_cdf(uniform_samples[:, i])
                
            return generated_samples
    
    elif emu_inputs_file is not None:
        LOGGER.info("using input file")
        with h5py.File(emu_inputs_file) as f: 
            sampled_values = f["emu_inputs"][:, :]
                
    else:
        dimension = len(parameters_dicts)

        rng = np.random.default_rng(args.seed)  # create a random generator with a seed
        lhd = qmc.LatinHypercube(d=dimension, seed=rng).random(n=n_gridpoints)
        if cov_file is not None and mu_file is not None:
            LOGGER.info("Sampling from multivariate gaussian distribution")
            LOGGER.info("Applying a covariance factor of {}".format(cov_factor))
            if clip_distributions:
                LOGGER.info("Cliping the distribution to take samples only up to 0")
            sampled_values = emu_utils.sample_from_hypercube(
                lhd,
                prior_ranges,
                dist="multivariate_gaussian",
                cov_file=cov_file,
                mu_file=mu_file,
                cov_factor=cov_factor,
                clip=clip_distributions,
            )

        else:
            LOGGER.info("Sampling from uniform distribution")
            sampled_values = emu_utils.sample_from_hypercube(lhd, prior_ranges)
    
        LOGGER.info(f"Actual number of sampled values: {sampled_values.shape[0]}")

    for index in indices:
        start_index = (index- ind_offset) * num_samps_per_index 
        end_index = start_index + num_samps_per_index

        if (nf_model_file is None) and (gc_dir is None):
            sub_samples = sampled_values[start_index:end_index]
        
        elif gc_dir is not None:
            rng_key = jax.random.PRNGKey(int(index))
            sub_samples = generate_new_samples(empirical_cdfs, mean_gaussian, cov_gaussian, num_samps_per_index, rng_key)

        else: 
            key, subkey = jax.random.split(jax.random.PRNGKey(int(index)))
            sub_samples = model.sample(subkey, num_samps_per_index, sigma_factor=nf_factor)

        output_dict = {}
        for dataset in datasets:
            output_dict[dataset] = []

        params_array = []

        bad_indices = []
        for sub_sample_ind in range(sub_samples.shape[0]):
            LOGGER.info(f"working on sample: {start_index+sub_sample_ind}")

            params = sub_samples[sub_sample_ind]
            # try: 
            outputs = computation_function(params, parameters_dicts)

            for i, dataset in enumerate(datasets):
                output_dict[dataset].append(outputs[i])
            
            params_array.append(params)

            # except Exception as e:
            #     print(f"sample {start_index+sub_sample_ind} failed with error {e}")
            #     print("params, parameters dicts", params, parameters_dicts)
            #     bad_indices.append(sub_sample_ind)
            #     continue
        
        print(f"bad indices for index {index} are {bad_indices}")
        params_array = np.array(params_array)

        np.savez(
            scratch + f"/processed_results_{index}.npz",
            **{dataset: np.array(output_dict[dataset]) for dataset in datasets},
            params=params_array,
        )

        yield index


def merge(indices, args):
    (
        num_samps_per_index,
        prior_ranges,
        scratch,
        parameters_dicts,
        datasets,
        n_gridpoints,
        computation_function,
        cov_file,
        mu_file,
        cov_factor,
        clip_distributions,
        ind_offset,
        nf_model_file,
        nf_factor,
        gc_dir,
        gc_factor,
        emu_inputs_file,
    ) = setup(args)

    print("HIII")

    with h5py.File(scratch + "/total_data.h5", "a") as hdf_file:
        for index in tqdm(indices):
            # Load the processed results for the current index from the npz file

            try: 
                npz_file_path = scratch + f"/processed_results_{index}.npz"
                with np.load(npz_file_path, mmap_mode="r") as data:
                    for dataset in datasets + ["params"]:
                        
                        if data[dataset][0].ndim != 1:
                            print("flattening array")
                            flattened_data = []
                            # Loop through each sample in the num_samples axis
                            for sample in data[dataset]:
                                # Flatten each sample and append to the list
                                flattened_sample = sample.flatten()
                                flattened_data.append(flattened_sample)
                            
                            flattened_data = np.array(flattened_data)

                            if dataset == "Ploopl":
                                print("where are the zeros here?")
                                print(np.where(flattened_data[0]==0))

                            data_ = flattened_data
                        else:
                            data_ = data[dataset]

                        emu_utils.update_or_create_dataset(dataset, data_, hdf_file)
            
            except Exception as e:
                print(f"failed to load index {index} with error {e}")
                continue
