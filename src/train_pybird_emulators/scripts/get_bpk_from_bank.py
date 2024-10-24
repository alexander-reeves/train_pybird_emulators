import numpy as np
import argparse
from classy import Class
import h5py
import os
from cosmic_toolbox import logger
from train_pybird_emulators.emu_utils import emu_utils
from pybird.correlator import Correlator    
from tqdm import tqdm
from filelock import Timeout, FileLock

'''
In this file we take in a hdf5 file containing some full bpk data already generated with `generate_data.py` and add a new data column
that contains the reconstructed bpks from the knot decomposition alone thereby providing a testbed for the knot decomoposition process.

'''

LOGGER = logger.get_logger(__name__)

def setup(args):

    parser = argparse.ArgumentParser(description='Create a bank of power spectra for a given cosmology')
    parser.add_argument('--spec_per_ind', type=int, default=500, help='Number of spectra to analyze per index')
    parser.add_argument('--lhc_bank_file', type=str, default='/cluster/work/refregier/alexree/local_packages/train_pybird_emulators/src/train_pybird_emulators/data/lhc_pk_bank/pk_bank_new.h5', help='Directory where full bpk and associated parameters have been saved')
    parser.add_argument('--scratch', type=str, default='/cluster/scratch/areeves/bpk_full_from_bank', help='Directory where temporary reconstructed bpk data will be saved')

    parser.add_argument(
        "--verbosity",
        default="warning",
        type=str,
        action="store",
        help="Verbosity level",
    )
    parser.add_argument('--resum', action='store_true', help='Whether or not to include IR resummation')
    parser.add_argument('--k_l', type=float, default=1e-4, help='The value of k_l to use for the computation of the pybird pieces training data')
    parser.add_argument('--k_r', type=float, default=0.7, help='The value of k_r to use for the computation of the pybird pieces training data')
    parser.add_argument('--n_k', type=int, default=1000, help='Number of k values to use')

    args = parser.parse_args(args)

    #load the hdf5 file 
    # set up a file lock to prevent multiple processes from trying to read the same file at the same time
    lock_path = args.lhc_bank_file + ".lock"
    lock = FileLock(lock_path, timeout=1000)
    with lock:
        with h5py.File(args.lhc_bank_file, "a") as hdf_file:
            lhc_pk_lin = hdf_file["pk_lin"][:]
            cosmo_params = hdf_file["params"][:]
    

    if not os.path.exists(args.scratch):
        os.makedirs(args.scratch)

    return args, lhc_pk_lin, cosmo_params

def main(indices, args):

    args, lhc_pk_lin, cosmo_params  = setup(args)
    

    resum = args.resum
    outdir = "/cluster/work/refregier/alexree/local_packages/pybird_emu/data/eftboss/out" #hardcoded path for now 
    with open(os.path.join(outdir, 'fit_boss_onesky_pk_wc_cmass_ngc_l0.dat')) as f: data_file = f.read()
    eft_params_str = data_file.split(', \n')[1].replace("# ", "")
    eft_params = {key: float(value) for key, value in (pair.split(': ') for pair in eft_params_str.split(', '))}

    param_names = ["omega_cdm", "omega_b", "h", "Omega_k", "n_s", "N_ur", "m_ncdm", "w0_fld", "z"]
    kk = np.logspace(np.log10(args.k_l), np.log10(args.k_r), args.n_k)

    for index in indices:
        bpk_tot = [] 
        for i in tqdm(range(index*args.spec_per_ind, (index+1)*args.spec_per_ind)):
            if i > lhc_pk_lin.shape[0]:
                LOGGER.warning(f"Index {i} is out of bounds, max index is {lhc_pk_lin.shape[0]}")

            pk_lin = lhc_pk_lin[i]
            cosmo_params_i = cosmo_params[i]
            M = Class() #did we make a mistake by not including deg_ncdm in the parameter list?
            M.set({
            "N_ncdm": 1,
            "T_ncdm": 0.71611,
            "Omega_Lambda": 0,
            })
            M.set({key:val for key, val in zip(param_names, cosmo_params_i) if key != "z"})
            M.compute()
            f = M.scale_independent_growth_factor_f(cosmo_params_i[-1])
            D = M.scale_independent_growth_factor(cosmo_params_i[-1])
            Omega0_m = M.Omega0_m()
            z=cosmo_params_i[-1]
            bpk_i = emu_utils.compute_1loop_bpk(pk_lin, f, D, z, Omega0_m, kk, eft_params, resum=args.resum)
            #flatten the bpk_i array
            bpk_i = bpk_i.flatten()
            bpk_tot.append(bpk_i)
        
        bpk_tot = np.array(bpk_tot)
        np.savez(args.scratch + f'/bpk_resum{resum}_{index}.npz',  **{f'bpk_resum_{args.resum}': bpk_tot})

        yield index

def merge(indices, args):
    args, lhc_pk_lin, cosmo_params  = setup(args)

    with h5py.File(args.lhc_bank_file, 'a') as hdf_file:
        for index in indices:
            # Load the processed results for the current index from the npz file
            npz_file_path = args.scratch + f'/bpk_resum{args.resum}_{index}.npz'
            with np.load(npz_file_path, mmap_mode='r') as data:
                datasets = [f'bpk_resum_{args.resum}']
                #delete the old datasets from hdf5 file if they exist
                if index == 0:
                    for dataset in datasets:
                        if dataset in hdf_file:
                            del hdf_file[dataset]
                    print("hddf file keys", hdf_file.keys())
                for dataset in datasets:
                    emu_utils.update_or_create_dataset(dataset, data[dataset], hdf_file=hdf_file)

