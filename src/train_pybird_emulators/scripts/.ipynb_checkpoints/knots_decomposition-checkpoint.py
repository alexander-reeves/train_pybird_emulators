import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from copy import copy, deepcopy
from scipy.interpolate import interp1d, make_interp_spline
import h5py
from time import time
from scipy.optimize import minimize
import pickle 
from scipy.special import legendre
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from classy import Class
from pybird.correlator import Correlator
import importlib, pybird
importlib.reload(pybird.correlator)
from pybird.correlator import Correlator
import argparse
#import utils
from train_pybird_emulators.emu_utils import emu_utils as utils


def to_Mpc_per_h(_pk, _kk, h):
    ilogpk_ = interp1d(np.log(_kk), np.log(_pk), fill_value='extrapolate')
    return np.exp(ilogpk_(np.log(_kk*h))) * h**3

# def load_pk_cosmo_dict(file_path, nbank, kk):
#     indices = np.random.randint(low=0,high=50000,size=nbank)
#     with open(file_path + 'lhc_cosmo_dict.pkl', 'rb') as f:
#         lhc_cosmo_dicts = pickle.load(f)
#     lhc_pk_lin = []
#     with h5py.File(file_path + 'lhc_pk_lins.h5', 'r') as f:
#         for i in indices:
#             print(i)
#             print(f[str(i)])
#             lhc_pk_lin.append(to_Mpc_per_h(f[str(i)][:], kk, lhc_cosmo_dicts[i]['h']))

#     #Append to array and convert units
#     lhc_pk_lin = np.array(lhc_pk_lin)

#     return lhc_cosmo_dicts, lhc_pk_lin, nbank, indices

def load_pk_cosmo_dict(file_path, nbank, kk):
    indices = np.random.randint(low=0,high=5000,size=nbank)
    with open(file_path + 'lhc_cosmo_dict.pkl', 'rb') as f:
        lhc_cosmo_dicts = pickle.load(f)
    lhc_pk_lin = []
    with h5py.File(file_path + 'lhc_pk_lins.h5', 'r') as f:
        keys=list(f.keys())
        for i in indices:
            #lhc_pk_lin.append(to_Mpc_per_h(f[str(i)][:], kk, lhc_cosmo_dicts[i]['h']))
            lhc_pk_lin.append(to_Mpc_per_h(f[keys[i]][:], kk, lhc_cosmo_dicts[i]['h']))


    #Append to array and convert units
    lhc_pk_lin = np.array(lhc_pk_lin)

    return lhc_cosmo_dicts, lhc_pk_lin, nbank, indices


def load_pk_cosmo_dict_new(file_path, nbank, kk):
    with h5py.File(file_path + 'total_data.h5', 'r') as f:
        lhc_pklin = f["pk_lin"][:]
        param_array = f["params"][:]

    lhc_pk_lin = np.array(lhc_pk_lin)

    param_order = ['omega_cdm', 'omega_b', 'h', 'Omega_k', 'z', 'sigma8', 'n_s']

    lhc_cosmo_dict = {
        param_order[i]: param_array[:, i] for i in range(param_array.shape[1])
    }

    return lhc_cosmo_dict, lhc_pk_lin, nbank

def get_spline_params(knots, ilogpk, with_logknots=False): 
    logknots = np.log( np.unique(knots) ) # sorting and removing duplicates 
    logpk = ilogpk(logknots)
    if with_logknots: return logknots, logpk
    else: return logpk
    
def _get_spline(logknots, logpk, kk, bc_type=None): 
    if bc_type: ilogpk_spline = make_interp_spline(logknots, logpk, k=3, bc_type=([(1, 1)], [(1, bc_type[1][0][1])]))
    else: ilogpk_spline = make_interp_spline(logknots, logpk, k=1)
    return np.exp(ilogpk_spline(np.log(kk)))

def get_spline(knots, ilogpk, kk, bc_type=None): 
    logknots, logpk = get_spline_params(knots, ilogpk, with_logknots=True)
    return _get_spline(logknots, logpk, bc_type=bc_type, kk=kk)


def get_logslope(x, f, side='left'):
    if side == 'left': 
        n = (np.log(f[1]) - np.log(f[0])) / (np.log(x[1]) - np.log(x[0]))
        A = f[0] / x[0]**n
    elif side == 'right':
        n = (np.log(f[-1]) - np.log(f[-2])) / (np.log(x[-1]) - np.log(x[-2]))
        A = f[-1] / x[-1]**n
    return A, n  

def minimize_spline_all(knots_start, lhc_pk_dict, kk_, k_l, k_r, what='ploop'):

    nbank = len(lhc_pk_dict)
    nbins = len(kk_)

    # Extract fixed knots
    leftmost_knot = knots_start[0]
    rightmost_knot = knots_start[-1]

    # Internal knots to be optimized
    internal_knots_start = knots_start[1:-1]

    if what == 'ploop':
        scaling = (kk_/0.4)**2 
    else:
        scaling = 1.

    def get_chi2(internal_knots):
        # Reconstruct full knot list
        knots = np.concatenate(([leftmost_knot], internal_knots, [rightmost_knot]))

    # Check if the knots are in increasing order
        if np.any(np.diff(knots) <= 0):
            return np.inf  # Return a high chi^2 if order is violated

        chi2 = 0.
        for d in lhc_pk_dict:      
            chi2 += np.sum((scaling * (d['pkk_'] - get_spline(knots, d['ilogpk'], kk_, d['bc_type']))/d['err'])**2) 

        return chi2/nbank/nbins

    method='Nelder-Mead'
    nk = len(internal_knots_start)
    m = minimize(get_chi2, internal_knots_start, method=method, bounds=nk*[(k_l, k_r)]) #, options={'maxiter': 1000})

    # Reconstruct the optimized knot list
    optimized_knots = np.concatenate(([leftmost_knot], m['x'], [rightmost_knot]))

    return optimized_knots, m['fun']


def setup(args):

    parser = argparse.ArgumentParser(description='Run knots optimization')
    parser.add_argument('--nknots', type=int, default=100)
    parser.add_argument('--nbank', type=int, default=100)

    args = parser.parse_args(args)

    nknots = args.nknots
    nbank = args.nbank


    k_l, k_r =  1e-4, 1.0 #In Mpc/h
    #kk = np.linspace(k_l, k_r, 500)
    kk = np.linspace(k_l, k_r, 200)
    mask = (kk < k_l) | (kk > k_r)
    kk_ = kk[~mask]
    dk = kk_[1:]-kk_[:-1]
    dk = np.concatenate(([dk[0]], dk))

    #Get covariance matrix at reference cosmology 
    z=0.5
    M = Class()
    cosmo = {'omega_b': 0.02235, 'omega_cdm': 0.120, 'h': 0.675, 'ln10^{10}A_s': 3.044, 'n_s': 0.965}
    M.set(cosmo)
    M.set({'output': 'mPk', 'P_k_max_h/Mpc': 1, 'z_max_pk': z})
    M.compute()
    pk_lin = np.array([M.pk_lin(k*M.h(), z)*M.h()**3 for k in kk]) # k in Mpc/h, pk in (Mpc/h)^3
    ipk_h = interp1d(kk, pk_lin, kind='cubic')

    Vs = 1.e11      # ideal volume (in [Mpc/h]^3), > 3 x better than DESI / Euclid (BOSS ~ 3e9, DESI ~ 3e10)
    nbar = 2e-2     # ideal nbar (for b1~2) (in [Mpc/h]^3), > 12 x better than DESI / Euclid
    cov = utils.get_cov(kk, ipk_h, 1., 0, mult=1, nbar=nbar, Vs=Vs)
    err = np.sqrt(np.diag(cov))

    lhc_cosmo_dicts, lhc_pk_lin, nbank, indices_pkbank = load_pk_cosmo_dict('../lhc_bank_z0p5to2p5/', nbank=nbank, kk=kk)
    lhc_pk_dict = []

    for i in range(indices_pkbank.shape[0]):

        pk_i = 1.*lhc_pk_lin[i]
        pk_max = np.max(pk_i)
        pk = 1.*pk_i / pk_max # normalizing

        ilogpk = interp1d(np.log(kk), np.log(pk), kind = 'cubic')
        pkk_ = np.exp(ilogpk(np.log(kk_)))


        A_l, n_l = get_logslope(kk_, pkk_, side='left')
        A_r, n_r = get_logslope(kk_, pkk_, side='right')

        pk_l = A_l * k_l**n_l
        pk_r = A_r * k_r**n_r

        dpk_l = A_l * n_l * k_l**(n_l-1.)
        dpk_r = A_r * n_r * k_r**(n_r-1.)

        dlogpk_l = (k_l/pk_l * dpk_l)
        dlogpk_r = (k_r/pk_r * dpk_r)

        bc_type = ([(1, dlogpk_l)], [(1, dlogpk_r)])

        lhc_pk_dict.append({'pkk_': pkk_, 'ilogpk': ilogpk, 'bc_type': bc_type, 'err': err, 'pk_max': pk_max}) 

    return nknots, k_l, k_r, kk_, lhc_pk_dict

def main(indices, args):
    nknots, k_l, k_r, kk_, lhc_pk_dict = setup(args)
    Vs = 1.e11      # ideal volume (in [Mpc/h]^3), > 3 x better than DESI / Euclid (BOSS ~ 3e9, DESI ~ 3e10)
    nbar = 5e-3     # ideal nbar (for b1~2) (in [Mpc/h]^3), > 3 x better than DESI / Euclid

    for index in indices: 
        random_seed = index
        # Define the initial knot list
        # try: 
        #     knots_start = np.load(f"knots_temp_file_knots_{nknots}.npy")

        # except FileNotFoundError:
        #random ICs for MonteCarlo search 

        # nlog = 55
        # nlog_lowk = 15
        nlog = 45
        nlog_lowk = 5
        knots_log_low_k = np.geomspace(k_l,1e-2, nlog_lowk, endpoint=False)
        knots_lin = np.linspace(1e-2, 0.2, nknots-nlog-nlog_lowk, endpoint=False) 
        knots_log = np.geomspace(0.2,k_r, nlog)
        


        # add random shifts for MonteCarlo approaxh 
        perturb_scale = 0.03
        knots_lin += np.random.uniform(-perturb_scale * knots_lin, perturb_scale * knots_lin, size=knots_lin.shape)
        knots_log_low_k[1:] *= np.exp(np.random.uniform(-perturb_scale, perturb_scale, size=knots_log_low_k[1:].shape))
        knots_log[:-1] *= np.exp(np.random.uniform(-perturb_scale, perturb_scale, size=knots_log[:-1].shape))

        knots_start = np.concatenate(([k_l],knots_log_low_k[1:], knots_lin, knots_log[:-1] , [k_r]))

        knots_start = np.clip(knots_start, k_l, k_r) #ensure within the k_l/k_r bounds
        # Optimize the knots

        knots=knots_start
        minchi2, delta_chi2 = 1e16, 1e16
        t0 = time()
        i = 0
        while delta_chi2 > 0.00001:
            knots, minchi2_i = minimize_spline_all(knots, lhc_pk_dict, kk_, k_l, k_r, what="ploop")
            ti = time()
            print ('step %s in %.1f sec, [red min chi2]^0.5=%.3f' % (i, ti-t0, minchi2_i**.5))
            delta_chi2 = minchi2 - minchi2_i
            minchi2 = 1. * minchi2_i
            t0 = 1.*ti
            i += 1

        np.save(f"./optimised_knots_cov_err/knots_temp_file_knots_{nknots}_index_{index}.npy", knots)

        print("finished in ", time()-t0, " seconds")

        yield index 