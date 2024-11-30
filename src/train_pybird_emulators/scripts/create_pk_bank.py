import numpy as np
import argparse
from classy import Class
from pyDOE import lhs
import h5py
import os
from cosmic_toolbox import logger
from train_pybird_emulators.emu_utils import emu_utils
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import uniform
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
from scipy.stats import qmc
from tqdm import tqdm
from pybird.correlator import Correlator
from train_pybird_emulators.emu_utils.emu_utils import get_pgg_from_linps_and_f_and_A



LOGGER = logger.get_logger(__name__)

def setup(args):

    parser = argparse.ArgumentParser(description='Create a bank of power spectra for a given cosmology')
    parser.add_argument('--n_pk', type=int, default=100, help='Number of power spectra to generate')
    parser.add_argument('--n_k', type=int, default=1000, help='Number of k values to use')
    parser.add_argument('--spec_per_ind', type=int, default=500, help='Number of spectra per index')
    parser.add_argument('--output_dir', type=str, default='pk_bank', help='Directory to save the power spectra')
    parser.add_argument('--k_l', type=float, default=1e-4, help='The value of k_l to use for the computation of the pybird pieces training data')
    parser.add_argument('--k_r', type=float, default=0.7, help='The value of k_r to use for the computation of the pybird pieces training data')
    parser.add_argument('--cov_file', type=str, default='/cluster/work/refregier/alexree/local_packages/train_pybird_emulators/src/train_pybird_emulators/data/lhc_pk_bank/gaussian_bank/cov_normal_mnu.npy', help='The file containing the covariance matrix for the parameters')
    parser.add_argument('--cov_factor', type=float, default=1.0, help='The factor to multiply the covariance matrix by')
    parser.add_argument("--clip_mode", default="pk_bank_train", type=str, action="store", help="Mode for clipping the samples")
    parser.add_argument("--make_pybird_training_data", default=False, action="store_true", help="Whether to make the pybird training data")
    parser.add_argument(
        "--verbosity",
        default="warning",
        type=str,
        action="store",
        help="Verbosity level",
    )
    args = parser.parse_args(args)

    # planck_mean = {'omega_b': 0.02235, 'omega_cdm': 0.120, 'h': 0.675, 'ln10^{10}A_s': 3.044, 'n_s': 0.965, 'Omega_k': 0., 'N_ncdm': 1., 'm_ncdm': 0.06, 'T_ncdm': 0.71611, 'N_ur': 2.0329, 'w0_fld': -1, 'Omega_Lambda': 0.}
    # lss_sigma = {'omega_b': 0.00035, 'omega_cdm': 0.010, 'h': 0.015, 'ln10^{10}A_s': 0.15, 'n_s': 0.060, 'w0_fld': 0.03, 'm_ncdm': 0.2, 'N_ur': 0.2, 'Omega_k': 0.05} #where is this from? 

    # #these are planck bestfit +- 5 lss sigma
    # param_ranges = {
    #     "omega_cdm": [planck_mean["omega_cdm"] - 5*lss_sigma["omega_cdm"], planck_mean["omega_cdm"]+5*lss_sigma["omega_cdm"]],   # omega_cdm
    #     "omega_b": [planck_mean["omega_b"] - 5*lss_sigma["omega_b"], planck_mean["omega_b"]+5*lss_sigma["omega_b"]], # omega_b
    #     "h": [planck_mean["h"]-5*lss_sigma["h"], planck_mean["h"] + 5* lss_sigma["h"]],   # h
    #     "Omega_k": [planck_mean["Omega_k"]- 5*lss_sigma["Omega_k"], planck_mean["Omega_k"] + 5*lss_sigma["Omega_k"]],    # curvature
    #     # "sigma8": [planck_mean["sigma8"]-5*lss_sigma["sigma8"], planck_mean["sigma8"]+5*lss_sigma["sigma8"]],   # sigma_8 #we dont not vary sigma8 as this is just a normalization parameter
    #     "n_s": [planck_mean["n_s"] - 5* lss_sigma["n_s"], planck_mean["n_s"] + 5* lss_sigma["n_s"]],    # n_s
    #     "N_ur":[planck_mean["N_ur"] - 5* lss_sigma["N_ur"], planck_mean["N_ur"] + 5* lss_sigma["N_ur"]], # N_ur
    #     "m_ncdm":[planck_mean["m_ncdm"] - 5* lss_sigma["m_ncdm"], planck_mean["m_ncdm"] + 5* lss_sigma["m_ncdm"]], # m_ncdm
    #     "w0_fld":[planck_mean["w0_fld"] - 5* lss_sigma["w0_fld"], planck_mean["w0_fld"] + 5* lss_sigma["w0_fld"]], # w0_fld
    #     "z":[0,4]
    # }

    # if output_dir does not exist, create it
    if not os.path.exists(args.output_dir):
        LOGGER.info(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir)

    print("output dir", args.output_dir)
    return args

def main(indices, args):

    args = setup(args)

    kk = np.logspace(np.log10(args.k_l), np.log10(args.k_r), args.n_k)

    # lhs_samples = lhs(n=len(param_ranges.keys()), samples=args.n_pk, criterion='center')

    # scaled_samples = {}
    # for i, key in enumerate(param_ranges.keys()):
    #     min_val, max_val = param_ranges[key]
    #     scaled_samples[key] = lhs_samples[:, i] * (max_val - min_val) + min_val

    # recently fixed the man with regrds to N_ur! SET Omega_m not omega_m !!!!!
    planck_mean = {'omega_b': 0.02235, 'Omega_m': 0.315, 'h': 0.675, 'ln10^{10}A_s': 3.044, 'n_s': 0.965, 'Omega_k': 0., 'm_ncdm': 0.02, 'N_ur': 0.00641}
    paramnames = ["ln10^{10}A_s", "h", "Omega_m", "m_ncdm", "n_s", "N_ur", "Omega_k","omega_b"] # order of parameters , m_nu is total neutrino mass
    mean_vector = np.array([planck_mean[key] for key in paramnames])
    dimension = len(paramnames)
    rng = np.random.default_rng(200)
    lhd = qmc.LatinHypercube(d=dimension, seed=rng).random(n=args.n_pk*35) #latin hypercube design (LHD) sampling takea buffer of 10X so we can remove negative m_nu
    sampled_values = emu_utils.sample_from_hypercube(
    lhd,
    prior_ranges=None, #these do nothing when used in gaussian mode so safer to leave them to None
    dist="multivariate_gaussian",
    cov_file=args.cov_file,
    mu_file=mean_vector,
    cov_factor=args.cov_factor,
    clip=True,
    clip_mode=args.clip_mode,
)   

    #check that the number of samplesis still bigger than the number of pk we want
    if sampled_values.shape[0] < args.n_pk:
        print("shape", sampled_values.shape[0])
        raise ValueError("Not enough samples after removing negative m_ncdm and ensuring N_ur is between 0 and 5. Try increasing the number of samples in the LHD design.")
    
    scaled_samples = {}
    for i, key in enumerate(paramnames):
        scaled_samples[key] = sampled_values[:, i]

    # finally add some redshift samples from a separte mini latin hypercube
    z_samples = qmc.LatinHypercube(d=1, seed=rng).random(n=args.n_pk)[:,0] * 4. #z goes from 0 to 4
    scaled_samples["z"] = z_samples

    #print one of the samples to check
    print(scaled_samples)
    
    outdir = "/cluster/work/refregier/alexree/local_packages/pybird_emu/data/eftboss/out" #hardcoded path for now 
    with open(os.path.join(outdir, 'fit_boss_onesky_pk_wc_cmass_ngc_l0.dat')) as f: data_file = f.read()
    eft_params_str = data_file.split(', \n')[1].replace("# ", "")
    eft_params = {key: float(value) for key, value in (pair.split(': ') for pair in eft_params_str.split(', '))}

    for index in indices: 

        pk = np.zeros((args.spec_per_ind, args.n_k))
        D = np.zeros((args.spec_per_ind,))
        f = np.zeros((args.spec_per_ind,))
        bpk_resum_True = np.zeros((args.spec_per_ind, 231)) #hardcode 231 for now
        bpk_resum_False = np.zeros((args.spec_per_ind, 231)) #hardcode 231 for now 
        emu_inputs = np.zeros((args.spec_per_ind, 82)) #hardcode 82 for now

        #pybird pieces
        Ploopl = np.zeros((args.spec_per_ind, 35*3*77))
        IRPs11 = np.zeros((args.spec_per_ind, 3*3*77))
        IRPsct = np.zeros((args.spec_per_ind, 6*3*77))
        IRPsloop = np.zeros((args.spec_per_ind, 35*3*77))



        subset = {key: scaled_samples[key][index*args.spec_per_ind:(index+1)*(args.spec_per_ind)] for key in scaled_samples.keys()}

        for i in tqdm(range(args.spec_per_ind)):
            #set up the class object
            try:
                cosmo = Class()
                #set the parameters
                cosmo.set({key: subset[key][i] for key in subset.keys() if key not in ['z']})
                #compute the power spectrum
                cosmo.set({"output":"mPk", 
                            "N_ncdm": 1,
                            "deg_ncdm": 3, #3 degenerate neutrino mass specied
                            "Omega_Lambda": 0,
                        "P_k_max_1/Mpc": 30.,
                        'z_max_pk': subset['z'][i]})
                cosmo.compute()
            
            except Exception as e:
                LOGGER.error(f"Error in computing power spectrum for index {index}, subset {i}: {e}")
                continue

            pk[i] = np.array([cosmo.pk_lin(k*cosmo.h(), subset["z"][i])*cosmo.h()**3 for k in kk])
            D[i] = cosmo.scale_independent_growth_factor(subset["z"][i])
            f[i] = cosmo.scale_independent_growth_factor_f(subset["z"][i])

            #whilst were here lets also compute some fiducial bpks for later testing so we have everything in the same bank 
            Omega0_m = cosmo.Omega0_m()
            z = subset["z"][i]
            bpk_resum_True[i] = emu_utils.compute_1loop_bpk(pk[i], f[i], D[i], z, Omega0_m, kk, eft_params, resum=True).flatten()
            bpk_resum_False[i] = emu_utils.compute_1loop_bpk(pk[i], f[i], D[i], z, Omega0_m, kk, eft_params, resum=False).flatten()

            # and finally lets get the emulator inputs
            knots = np.load("/cluster/work/refregier/alexree/local_packages/pybird_emu/data/emu/knots.npy")
            # slightly modify the start and end knot to fall in the range of k we are using so that the interpolation does not fail
            eps = 1e-10
            knots[0] = knots[0] + eps
            knots[-1] = knots[-1] - eps
            param_realization = [] 
            pk_i = 1.*pk[i]
            f_i = 1.*f[i]
            pk_max = np.max(pk_i)
            pk_norm = 1.*pk_i / pk_max # normalizing
            #Get the spline params
            ilogpk = interp1d(np.log(kk), np.log(pk_norm), kind = 'cubic')
            spline_params = emu_utils.get_spline_params(knots,ilogpk)
            param_realization.extend(list(spline_params))
            param_realization.append(pk_max)
            param_realization.append(f_i)
            emu_inputs_i = np.array(param_realization)
            emu_inputs[i] = emu_inputs_i

            # finaly whilst were here lets just egt the pieces needed for the training as well! We can make this more accurate by using the actual Pklin inputs!
            if args.make_pybird_training_data:
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
                _, _, Ploopl_i, IRPs11_i, IRPsct_i, IRPsloop_i = get_pgg_from_linps_and_f_and_A(pk_norm, kk=kk, N=N, f=f[i], A=pk_max)
                Ploopl[i] = Ploopl_i
                IRPs11[i] = IRPs11_i
                IRPsct[i] = IRPsct_i
                IRPsloop[i] = IRPsloop_i


        
        #turn all of the parameter values stored in the subset dict into a numpy array
        param_values = np.array([subset[key] for key in subset.keys()]).T
        dtype = param_values.dtype
        #transform param_values to a structured array
        param_values = np.array([tuple(row) for row in param_values], dtype=[(key, dtype) for key in subset.keys()])

        np.savez(args.output_dir + "/pk_" + str(index), pk_lin=pk, params=param_values, kk=kk, D=D, f=f, bpk_resum_True=bpk_resum_True, bpk_resum_False=bpk_resum_False, emu_inputs=emu_inputs,\
        Ploopl=Ploopl, IRPs11=IRPs11, IRPsct=IRPsct, IRPsloop=IRPsloop)
        LOGGER.info(f"Saved power spectra and associate bpk for index: {index}")
    
    yield indices 

def merge(indices, args):
    print("Starting merge function")
    args = setup(args)
    with h5py.File(args.output_dir + '/total_data.h5', 'a') as hdf_file:
        for index in indices:
            print("hello")
            # Load the processed results for the current index from the npz file
            npz_file_path = args.output_dir + f'/pk_{index}.npz'

            try:
                LOGGER.info(f"Pk_lin shape: {np.load(npz_file_path)['pk_lin'].shape}")
                LOGGER.info(f"params shape: {np.load(npz_file_path)['params'].shape}")
                with np.load(npz_file_path, mmap_mode='r') as data:
                    datasets = ["pk_lin", "params", "kk", "D", "f", "bpk_resum_True", "bpk_resum_False", "emu_inputs", "Ploopl", "IRPs11", "IRPsct", "IRPsloop"]

                    for dataset in datasets:
                        emu_utils.update_or_create_dataset(dataset, data[dataset], hdf_file=hdf_file)

            except Exception as e:
                print(f"could not find file for index {index}") 
