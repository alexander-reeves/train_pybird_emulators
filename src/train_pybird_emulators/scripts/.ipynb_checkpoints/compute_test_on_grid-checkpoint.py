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
from classy import Class

def setup(args):
    '''
    In the set-up we load in the power spectra to iterature over 
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dicts_path", type=str, required=True)
    parser.add_argument("--num_per_ind", type=int, required=True)
    parser.add_argument("--filename_knots", type=str, required=True, help='optimal k-knots placement file as a .npy')
    parser.add_argument("--resum", action='store_true', default=False, help='if true, will perform IR resummation')

    args = parser.parse_args(args)

    output_fn = '/cluster/scratch/areeves/ff_linearps_test/'

    input_dicts_path = args.input_dicts_path
    num_per_ind = args.num_per_ind
    filename_knots = args.filename_knots
    resum = args.resum

    knots = np.load(filename_knots)

    kk = np.logspace(-5, 0, 200)


    with open("/cluster/work/refregier/alexree/frequentist_framework/FreqCosmo/hpc_work/eft_params_boss_cmass_ngc_l0", 'rb') as f:
        eft_params = pickle.load(f)

    return output_fn, eft_params, kk, input_dicts_path, num_per_ind, knots, resum


# Function to load and save data to HDF5
def watchdog(indices, args):
    # Create the HDF5 database
    output_fn, eft_params, kk, input_dicts_path, num_per_ind, knots, resum = setup(args)
    nknots = knots.shape[0]

    if not resum: 
        out_fn = f'/cluster/scratch/areeves/ff_linearps_test/processed_test_data_mpk_bpk_{nknots}_knots_noresum.h5'
    else: 
        out_fn = f'/cluster/scratch/areeves/ff_linearps_test/processed_test_data_mpk_bpk_{nknots}_knots.h5'

    with h5py.File(out_fn, 'w') as hdf_file:
        for index in indices:
            # Load the processed results for the current index from the npz file
            for i in range(index*num_per_ind, (index+1)*(num_per_ind)):
                
                if resum: 
                    npz_file_path = output_fn + f"/testing_data_bpk_{nknots}_knots{i}.npz"
                
                else: 
                    npz_file_path = output_fn + f"/testing_data_bpk_{nknots}_knots{i}_noresum.npz"
                with np.load(npz_file_path) as data:
                    bpk_true = data['bpk_true']
                    bpk_emu = data['bpk_emu']


                # Create a subgroup for the current index if it doesn't exist
                subgroup = hdf_file.require_group(str(i))
                subgroup.create_dataset('bpk_true', data=bpk_true)
                subgroup.create_dataset('bpk_emu', data=bpk_emu)


def main(indices, args): 

    output_fn, eft_params, kk, input_dicts_path, num_per_ind, knots, resum  = setup(args)

    nknots = knots.shape[0]
    with open(input_dicts_path, 'rb') as file:
        # Load the dictionary from the file
        dict_list = pickle.load(file)

    for index in indices:
        for i in range(index*num_per_ind, (index+1)*(num_per_ind)):
            if i<5000:
    
                cosmo_dict = dict_list[i]
                bpk_true, bpk_emu = utils.compute_1loop_bpk_test(cosmo_dict, kk, eft_params, knots, resum=resum)

                if resum: 
                    np.savez(output_fn + f"/testing_data_bpk_{nknots}_knots{i}", bpk_true=bpk_true, bpk_emu=bpk_emu)
                else:
                    np.savez(output_fn + f"/testing_data_bpk_{nknots}_knots{i}_noresum", bpk_true=bpk_true, bpk_emu=bpk_emu)

            else: 
                print("i greater that 5000, quitting...")
                break 
        
        yield index