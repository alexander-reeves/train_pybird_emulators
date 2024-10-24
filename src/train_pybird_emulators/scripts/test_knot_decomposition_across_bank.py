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
    parser.add_argument('--full_bpk_hdf5_file', type=str, default='/cluster/scratch/areeves/pybird_training_data_bpk_full4/total_data.h5', help='Directory where full bpk and associated parameters have been saved')
    parser.add_argument('--knots_file', type=str, default='/cluster/work/refregier/alexree/local_packages/pybird_emu/data/emu/knots.npy', help='Directory where the knots have been saved')
    parser.add_argument('--scratch', type=str, default='/cluster/scratch/areeves/bpk_knots_reconstructed_2', help='Directory where temporary reconstructed bpk data will be saved')

    parser.add_argument(
        "--verbosity",
        default="warning",
        type=str,
        action="store",
        help="Verbosity level",
    )
    args = parser.parse_args(args)

    #load the hdf5 file 
    # set up a file lock to prevent multiple processes from trying to read the same file at the same time
    lock_path = args.full_bpk_hdf5_file + ".lock"
    lock = FileLock(lock_path, timeout=50)
    with lock:
        with h5py.File(args.full_bpk_hdf5_file, "a") as hdf_file:
            bpk_array = np.array(hdf_file["bpk"][:])
            pk_array = np.array(hdf_file["pk_lin"][:])
            f1_array = np.array(hdf_file["f1"][:])
            D1_array = np.array(hdf_file["D1"][:])
            params_array = np.array(hdf_file["params"][:])
            kk_array = np.array(hdf_file["kk"][:])
    

    if not os.path.exists(args.scratch):
        os.makedirs(args.scratch)

    return args, bpk_array, pk_array, f1_array, D1_array, params_array, kk_array

def main(indices, args):

    args, bpk_array, pk_array, f1_array, D1_array, params_array, kk_array  = setup(args)
    
    LOGGER.info(f"Loaded data from {args.full_bpk_hdf5_file}")
    LOGGER.info(f"Loaded data with shape: bpk_array: {bpk_array.shape}")

    #create a correlator engine instance for cimputing the bpk 
    resum = True
    N = Correlator()
    N.set(      {"output": "bPk",
                "multipole": 3,
                "kmax": 0.4,
                "fftaccboost": 2,  # boosting the FFTLog precision (slower, but ~0.1% more precise -> let's emulate this)
                "with_resum": resum,
                "with_exact_time": True,
                "km": 1.0,
                "kr": 1.0,
                "nd": 3e-4,
                "with_emu":False,
                "eft_basis": "eftoflss",
                "with_stoch": True,})
    
    knots = np.load(args.knots_file)
    outdir = "/cluster/work/refregier/alexree/local_packages/pybird_emu/data/eftboss/out" #hardcoded path for now 
    with open(os.path.join(outdir, 'fit_boss_onesky_pk_wc_cmass_ngc_l0.dat')) as f: data_file = f.read()
    eft_params_str = data_file.split(', \n')[1].replace("# ", "")
    eft_params = {key: float(value) for key, value in (pair.split(': ') for pair in eft_params_str.split(', '))}

    for index in indices:
        bpk_tot = [] 
        for i in tqdm(range(index*args.spec_per_ind, (index+1)*args.spec_per_ind)):
            if i > bpk_array.shape[0]:
                LOGGER.warning(f"Index {i} is out of bounds, max index is {bpk_array.shape[0]}")

            pk_lin = pk_array[i]
            f1 = f1_array[i]
            D1 = D1_array[i]
            params = params_array[i]
            kk = kk_array[0]
            bpk_i = emu_utils.get_bpk_spline_fit(pk_lin, f1, D1, params, kk, N, eft_params, knots, resum=True)
            #flatten the bpk_i array
            bpk_i = bpk_i.flatten()
            bpk_tot.append(bpk_i)
        
        bpk_tot = np.array(bpk_tot)
        np.savez(args.scratch + f'/bpk_knots_{index}.npz', bpk_knots_reconstructed=bpk_tot)

        yield index

def merge(indices, args):
    args, bpk_array, pk_array, f1_array, D1_array, params_array, kk_array = setup(args)
    with h5py.File(args.full_bpk_hdf5_file, 'a') as hdf_file:
        for index in indices:
            # Load the processed results for the current index from the npz file
            npz_file_path = args.scratch + f'/bpk_knots_{index}.npz'
            with np.load(npz_file_path, mmap_mode='r') as data:
                datasets = ["bpk_knots_reconstructed"]
                #delete the old datasets from hdf5 file if they exist
                if index == 0:
                    for dataset in datasets:
                        if dataset in hdf_file:
                            del hdf_file[dataset]
                for dataset in datasets:
                    emu_utils.update_or_create_dataset(dataset, data[dataset], hdf_file=hdf_file)

