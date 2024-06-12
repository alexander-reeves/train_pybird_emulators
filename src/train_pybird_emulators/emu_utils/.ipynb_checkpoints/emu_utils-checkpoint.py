import yaml
import numpy as np
from scipy.stats import qmc, norm, truncnorm, uniform
import pybird
from pybird.correlator import Correlator
from pybird.fftlog import FFTLog
from classy import Class
from scipy.interpolate import interp1d, make_interp_spline
from scipy.sparse import csr_matrix
from scipy.stats import norm
from scipy.special import legendre
import jax.numpy as jnp
import jax
from cosmic_toolbox import logger

def read_yaml_file(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data


def sample_from_hypercube(lhc, prior_ranges, dist="uniform", cov_file=None, mu_file=None):
    num_samples, num_params = lhc.shape
    sampled_values = np.zeros((num_samples, num_params))

    if dist == "uniform":
        for i in range(num_params):
            lower_bound, upper_bound = prior_ranges[i]
            parameter_range = upper_bound - lower_bound
            sampled_values[:, i] = (lhc[:, i] * parameter_range) + lower_bound

        return sampled_values

    elif dist == "gaussian":
        for i in range(num_params):
            mu, sigma = prior_ranges[i]
            sampled_values[:, i] = norm.ppf(lhc[:, i], loc=mu, scale=sigma)

        return sampled_values

    elif dist == "multivariate_gaussian":
        mu_vector, covariance_matrix = np.load(mu_file), np.load(cov_file)   
        cholesky_decomposition = np.linalg.cholesky(covariance_matrix)

        # Transform LHC samples using inverse of univariate Gaussian CDF
        transformed_lhc = norm.ppf(lhc)
        print(mu_vector.shape)
        print(transformed_lhc.shape)
        print(cholesky_decomposition.shape)
        #sampled_values = mu_vector + transformed_lhc @ cholesky_decomposition.T
        sampled_values = mu_vector + transformed_lhc.T @ cholesky_decomposition

        return sampled_values


class PiecewiseSpline_jax: 
    def __init__(self, knots, logpk_knots, dlogpk_r=1, boundaries=False):
        #dummy dlogpkr value if none supplied 
        self.knots = knots
        self.logknots = jnp.log(knots)
        self.logpk_knots = logpk_knots

        # Store boundary values and their derivatives
        self.k_l = self.knots[0]
        self.k_r = self.knots[-1]

        self.nr = dlogpk_r
        self.logAr = self.logpk_knots[-1] - self.nr*self.logknots[-1]
        self.logAl = self.logpk_knots[0] - 1.*self.logknots[0]

        # Calculate the gradient at the first knot for left extrapolation
        # self.gradient_left = (logpk_knots[1] - logpk_knots[0]) / (self.logknots[1] - self.logknots[0])
        self.gradient_left = 1 #set this to be 1 always

        #use JAX spline 
        if boundaries: 
            p1_l = self.logAl + 1 * jnp.log(self.k_l*0.999) 
            p2_l = self.logAl + 1 * jnp.log(self.k_l*0.998) 
            p3_l = self.logAl + 1 * jnp.log(self.k_l*0.997)

            p1_r = self.logAr + self.nr * jnp.log(self.k_r*1.001) 
            p2_r = self.logAr + self.nr * jnp.log(self.k_r*1.002) 
            p3_r = self.logAr + self.nr * jnp.log(self.k_r*1.003)

            self.logknots = jnp.concatenate((jnp.log(jnp.array([self.k_l*0.997,self.k_l*0.998,self.k_l*0.999])), self.logknots, jnp.log(jnp.array([self.k_r*1.001,self.k_r*1.002,self.k_r*1.003]))))
            self.logpk_knots = jnp.concatenate((jnp.array([p3_l,p2_l,p1_l]), self.logpk_knots, jnp.array([p1_r,p2_r,p3_r])))
            print("logknots", self.logknots)
            self.ilogpk_spline = InterpolatedUnivariateSpline_jax(self.logknots, self.logpk_knots, k=3, endpoints='natural')

        else:
            self.ilogpk_spline = InterpolatedUnivariateSpline_jax(self.logknots, self.logpk_knots, k=3, endpoints='natural')

    def __call__(self, k):
        #vectorize the function over k using Jax
        return jax.vmap(self.evaluate)(k)

    def evaluate(self, logk):
        k = jnp.exp(logk)
        # Use jnp.where to handle conditional logic
        return jnp.where(
            k < self.k_l,
            self.logAl + 1.*logk,
            jnp.where(
                k > self.k_r,
                self.logAr + self.nr * logk,
                self.ilogpk_spline(logk)
            )
        )

class PiecewiseSpline:
    """
    A class that when called will give us the piecewise spline with modified left-hand side behavior.
    """
    def __init__(self, knots, logpk_knots, dlogpk_r):
        # self.knots = sorted(knots)
        self.knots = knots
        self.logknots = np.log(self.knots)
        self.logpk_knots = logpk_knots

        # Define the right boundary condition only
        self.bc = ([(1, 1)], [(1, dlogpk_r)])

        # Create the spline
        self.ilogpk_spline = make_interp_spline(self.logknots, self.logpk_knots, k=3, bc_type=self.bc)

        # Store boundary values and their derivatives
        self.k_l = self.knots[0]
        self.k_r = self.knots[-1]

        self.nr = dlogpk_r
        self.logAr = self.logpk_knots[-1] - self.nr*self.logknots[-1]
        self.logAl = self.logpk_knots[0] - 1.*self.logknots[0]

        # Calculate the gradient at the first knot for left extrapolation
        # self.gradient_left = (logpk_knots[1] - logpk_knots[0]) / (self.logknots[1] - self.logknots[0])
        self.gradient_left = 1 #set this to be 1 always

    def __call__(self, k):
        return np.vectorize(self.evaluate)(k)

    def evaluate(self, logk):
        k = np.exp(logk)
        if k < self.k_l:
            return self.logAl + 1.*logk
        elif k > self.k_r:
            return self.logAr + self.nr * logk
        else:
            return self.ilogpk_spline(logk)


def get_pgg_from_linps(pk_lin, kk, eft_params, N):
    '''
    Function to compute the galaxy redshift space multipoles given an inputs linear matter power spectrum
    '''

    #HMM this is the bit we should ask about-- there is no "true" plin in this case
    N.compute({'kk': kk, 'pk_lin': pk_lin,             # pk_lin normalized by A
            'pk_lin_2': pk_lin,                     # pk_lin_2 goes inside the loop integrals (+IR corr.)
            'D': 1., 'f': 1., 'z': 1., 'Omega0_m': 1.}, # a bunch of unused dummy values
            do_core=True, do_survey_specific=False)      # this step computes the following:

    outputs = [N.bird.P11l.astype(np.float32), N.bird.Pctl.astype(np.float32), N.bird.Ploopl.astype(np.float32),
       N.bird.IRPs11.astype(np.float32), N.bird.IRPsct.astype(np.float32), N.bird.IRPsloop.astype(np.float32)]

    return outputs

def get_pgg_from_linps_and_f_and_A(pk_lin, kk, eft_params, N, f, A):
    '''
    Function to compute the galaxy redshift space multipoles given an inputs linear matter power spectrum
    '''

    #HMM this is the bit we should ask about-- there is no "true" plin in this case
    N.compute({'kk': kk, 'pk_lin': pk_lin,             # pk_lin normalized by A
            'pk_lin_2': pk_lin,                     # pk_lin_2 goes inside the loop integrals (+IR corr.)
            'D': 1., 'f': 1., 'z': 1., 'Omega0_m': 1.}, # a bunch of unused dummy values
            do_core=True, do_survey_specific=False)      # this step computes the following:

    #Do the set resum part
    Q = N.resum.makeQ(f)
    Dp2 = A
    Dp2n = np.concatenate((2*[N.co.Na*[Dp2**(n+1)] for n in range(N.co.NIR)])) #function from PyBird but

    N.bird.IRPs11 = np.einsum('n,lnk->lnk', Dp2n, N.bird.IRPs11)
    N.bird.IRPsct = np.einsum('n,lnk->lnk', Dp2n, N.bird.IRPsct)
    N.bird.IRPsloop = np.einsum('n,lmnk->lmnk', Dp2n, N.bird.IRPsloop)

    N.bird.setIRPs(Q=Q)

    #Still keep the pieces separate as we might not want to resum
    outputs = [N.bird.P11l.astype(np.float32), N.bird.Pctl.astype(np.float32), N.bird.Ploopl.astype(np.float32),
       N.bird.fullIRPs11.astype(np.float32), N.bird.fullIRPsct.astype(np.float32), N.bird.fullIRPsloop.astype(np.float32)]

    return outputs


def to_Mpc_per_h(_pk, _kk, h):
    ilogpk_ = interp1d(np.log(_kk), np.log(_pk), fill_value='extrapolate')
    return np.exp(ilogpk_(np.log(_kk*h))) * h**3


def get_logslope(x, f, side='left'):
    if side == 'left': 
        n = (np.log(f[1]) - np.log(f[0])) / (np.log(x[1]) - np.log(x[0]))
        A = f[0] / x[0]**n
    elif side == 'right':
        n = (np.log(f[-1]) - np.log(f[-2])) / (np.log(x[-1]) - np.log(x[-2]))
        A = f[-1] / x[-1]**n
    return A, n

def compute_1loop_bpk_test(cosmo_dict, kk, eft_params, knots, resum=False):
    '''
    Function to compute the difference between the reconstructed and original 1-loop biased tracer power spectrum 
    given an input cosmology dictionary and optimal knots
    '''

    #hardcoded for now
    k_l, k_r =  1e-4, 1.0 #In Mpc/h 
    kk_ = np.linspace(k_l, k_r, 100)

    z = cosmo_dict['z']

    # knots = sorted(knots)
    logknots = np.log(knots)

    #Maybe keep this outside function for speed 
    N = Correlator()

    if resum:
        N.set({'output': 'bPk', 'multipole': 3, 'kmax': 0.4,
            'fftaccboost': 2, # boosting the FFTLog precision (slower, but ~0.1% more precise -> let's emulate this)
            'with_resum': True, 'with_exact_time': True,
            'km': 1., 'kr': 1., 'nd': 3e-4,
            'eft_basis': 'eftoflss', 'with_stoch': True})

    else: 
        N.set({'output': 'bPk', 'multipole': 3, 'kmax': 0.4,
            'fftaccboost': 2, # boosting the FFTLog precision (slower, but ~0.1% more precise -> let's emulate this)
            'with_resum': False, 'with_exact_time': True,
            'km': 1., 'kr': 1., 'nd': 3e-4,
            'eft_basis': 'eftoflss', 'with_stoch': True})

    #First use class to compute the linear growth factor and the f growth factor
    M = Class()
    keys_to_exclude = ["z", "pk_max"]
    cosmo = {k: v for k, v in cosmo_dict.items() if k not in keys_to_exclude}
    M.set(cosmo)
    M.set({'output': 'mPk', 'P_k_max_h/Mpc': 5, 'z_max_pk': cosmo_dict['z']})
    M.compute()
    pk_class = np.array([M.pk_lin(k*M.h(), cosmo_dict['z'])*M.h()**3 for k in kk]) # k in Mpc/h, pk in (Mpc/h)^3

    #Linear growth factor and other pieces required
    A_s, Omega0_m = 1e-10 * np.exp(cosmo['ln10^{10}A_s']), M.Omega0_m()
    A = np.max(pk_class)
    D1, f1 = M.scale_independent_growth_factor(cosmo_dict['z']), M.scale_independent_growth_factor_f(cosmo_dict['z']),

    N.compute({'kk': kk, 'pk_lin': pk_class, 'D': D1, 'f': f1, 'z': z, 'Omega0_m': Omega0_m},
            do_core=True, do_survey_specific=True)

    bpk_true = N.get(eft_params)

    #Now use the emulator for the same thing
    N2 = Correlator()

    if resum:
        N2.set({'output': 'bPk', 'multipole': 3, 'kmax': 0.4,
            'fftaccboost': 2, # boosting the FFTLog precision (slower, but ~0.1% more precise -> let's emulate this)
            'with_resum': True, 'with_exact_time': True,
            'with_time': False, # time unspecified
            'km': 1., 'kr': 1., 'nd': 3e-4,
            'eft_basis': 'eftoflss', 'with_stoch': True,
            # "with_uvmatch_2": True,
            'with_emu':True, 'emu_path': '/cluster/work/refregier/alexree/frequentist_framework/FreqCosmo/hpc_work/saved_models/stronger_cuts',
            'knots_path': '/cluster/work/refregier/alexree/frequentist_framework/FreqCosmo/hpc_work/final_knots_55.npy'
            })

    else:
        N2.set({'output': 'bPk', 'multipole': 3, 'kmax': 0.4,
            'fftaccboost': 2, # boosting the FFTLog precision (slower, but ~0.1% more precise -> let's emulate this)
            'with_resum': False, 'with_exact_time': True,
            'with_time': False, # time unspecified
            'km': 1., 'kr': 1., 'nd': 3e-4,
            'eft_basis': 'eftoflss', 'with_stoch': True,
            # "with_uvmatch_2": True,
            'with_emu':True, 'emu_path': '/cluster/work/refregier/alexree/frequentist_framework/FreqCosmo/hpc_work/saved_models/stronger_cuts',
            'knots_path': '/cluster/work/refregier/alexree/frequentist_framework/FreqCosmo/hpc_work/final_knots_55.npy'
            })

    N2.compute({'kk': kk, 'pk_lin': pk_class, 'pk_lin_2':pk_class, 'D': D1, 'f': f1, 'z': z, 'Omega0_m': Omega0_m}, 
            do_core=True, do_survey_specific=False, correlator_engine=True)

    N2.compute({'D': 1., 'f': f1, 'z': z, 'Omega0_m': Omega0_m, 'A': A, 'kmax':0.4},
            do_core=False, do_survey_specific=True,correlator_engine=True)

    bpk_emu = N2.get(eft_params)

    print(bpk_true/bpk_emu)

    return bpk_true, bpk_emu




def compute_1loop_mps_test(cosmo_dict, kk, eft_params, knots):
    '''
    Function to compute the difference between the reconstructed and original 1-loop matter power given
    an input cosmology dictionary and optimal knots
    '''
    #hardcoded for now
    k_l, k_r =  1e-4, 1.0 #In Mpc/h 
    kk_ = np.linspace(k_l, k_r, 100)

    knots = sorted(knots)
    logknots = np.log(knots)

    #Maybe keep this outside function for speed this sets up for real-space mps
    N = Correlator()

    pybird_config = {'output': 'mPk', 'multipole': 3, 'kmax': 0.6,
        'km': 0.7, 'kr': 0.35, 'nd': 1e-2,            # these scales control the various EFT expansions...
        'eft_basis': 'eftoflss', 'with_stoch': False, # there are various equivalent EFT parametrization one can choose
        'with_resum': True,                         #####
        'fftbias': -1.6,
        }

    N.set(pybird_config)

    #First use class to compute the linear growth factor and the f growth factor
    M = Class()
    keys_to_exclude = ["z", "pk_max"]
    cosmo = {k: v for k, v in cosmo_dict.items() if k not in keys_to_exclude}
    M.set(cosmo)
    M.set({'output': 'mPk', 'P_k_max_h/Mpc': 5, 'z_max_pk': cosmo_dict['z']})
    M.compute()
    pk_class = np.array([M.pk_lin(k*M.h(), cosmo_dict['z'])*M.h()**3 for k in kk]) # k in Mpc/h, pk in (Mpc/h)^3

    #Make the spline for the linear MPS 
    pk_max = np.max(pk_class)
    ilogpk = interp1d(np.log(kk), np.log(pk_class/pk_max), kind = 'cubic') #normalized interpolation

    logpk_knots = ilogpk(logknots)

    pkk_ = np.exp(ilogpk(np.log(kk_)))
    
    A_l, n_l = get_logslope(kk_, pkk_, side='left')
    A_r, n_r = get_logslope(kk_, pkk_, side='right')

    pk_l = A_l * k_l**n_l
    pk_r = A_r * k_r**n_r
    
    dpk_l = A_l * n_l * k_l**(n_l-1.)
    dpk_r = A_r * n_r * k_r**(n_r-1.)
    
    dlogpk_l = (k_l/pk_l * dpk_l) 
    dlogpk_r = (k_r/pk_r * dpk_r) 

    ilogpk_spline = PiecewiseSpline(knots, logpk_knots, dlogpk_l, dlogpk_r)

    #recover the linear matter power spectrum from spline 
    pk_lin_rec = np.exp(ilogpk_spline(np.log(kk)))*pk_max 

    #Finally match up the tails in k_l or k_r
    mask = (kk < k_l) | (kk > k_r)
    pk_class[mask] = pk_lin_rec[mask]

    #Linear growth factor and other pieces required
    A_s, Omega0_m = 1e-10 * np.exp(cosmo_dict['ln10^{10}A_s']), M.Omega0_m()
    D1, f1 = M.scale_independent_growth_factor(cosmo_dict['z']), M.scale_independent_growth_factor_f(cosmo_dict['z']),

    N.compute({'kk': kk, 'pk_lin': pk_class, 'pk_lin_2': pk_class,
           'D': D1, 'f': f1, 'z': cosmo_dict['z'], 'Omega0_m': Omega0_m, 'A': 1.},
          do_core=True, do_survey_specific=True)

    mpk_orig = N.get(eft_params)


    N.compute({'kk': kk, 'pk_lin': pk_class, 'pk_lin_2': pk_lin_rec,
            'D': D1, 'f': f1, 'z': cosmo_dict['z'], 'Omega0_m': Omega0_m, 'A': 1.},
            do_core=True, do_survey_specific=True)

    mpk_rec = N.get(eft_params)

    return mpk_orig, mpk_rec


def test_1loop_sparsity(cosmo_dict, kk, eft_params):
    '''
    Function to compute the difference between the reconstructed and original 1-loop matter power given
    an input cosmology dictionary and optimal knots
    '''
    #hardcoded for now
    k_l, k_r =  1e-4, 1.0 #In Mpc/h 
    #Maybe keep this outside function for speed this sets up for real-space mps
    N = Correlator()

    N.set({'output': 'bPk', 'multipole': 3, 'kmax': 0.6,
        'fftaccboost': 2, # boosting the FFTLog precision (slower, but ~0.1% more precise -> let's emulate this)
        'with_resum': True, 'with_exact_time': True,
        'with_time': False, # time unspecified
        'km': 1., 'kr': 1., 'nd': 3e-4,
        'eft_basis': 'eftoflss', 'with_stoch': True})

    #First use class to compute the linear growth factor and the f growth factor
    M = Class()
    keys_to_exclude = ["z", "pk_max"]
    cosmo = {k: v for k, v in cosmo_dict.items() if k not in keys_to_exclude}
    M.set(cosmo)
    M.set({'output': 'mPk', 'P_k_max_h/Mpc': 5, 'z_max_pk': cosmo_dict['z']})
    M.compute()
    pk_class = np.array([M.pk_lin(k*M.h(), cosmo_dict['z'])*M.h()**3 for k in kk]) # k in Mpc/h, pk in (Mpc/h)^3

    #Linear growth factor and other pieces required
    A_s, Omega0_m = 1e-10 * np.exp(cosmo_dict['ln10^{10}A_s']), M.Omega0_m()
    D1, f1 = M.scale_independent_growth_factor(cosmo_dict['z']), M.scale_independent_growth_factor_f(cosmo_dict['z']),

    N.compute({'kk': kk, 'pk_lin': pk_class, 'pk_lin_2': pk_class,
           'D': D1, 'f': f1, 'z': cosmo_dict['z'], 'Omega0_m': Omega0_m},
          do_core=True, do_survey_specific=True)

    bpk_0 = N.get(eft_params) #original bpk to compare size with 

    #Now loop through and find out the indices of small values of the IRloops
    Q = 1. * N.bird.Q
    M0 = 1. * N.bird.IRPsloop

    for p in range(M0.shape[0]):
        for i in range(M0.shape[1]):
            for n in range(M0.shape[2]):
                if (Q[1,:,p,n] == 0).all(): # if they are multiplied by 0, setting them to 0
                    M0[p,i,n] = 0.

    sparsity = 1 - np.count_nonzero(M0) / M0.size
    print("sparsity")
    print ('%.3f' % sparsity)

    C = csr_matrix(M0.reshape(-1))  # this is the sparse array stored in a compressed format
    # print (C.shape)
    C_idx = C.indices #non-zero indices

    return C_idx



def get_cov(kk, ipklin, b1, f1, Vs, nbar=3.e-4, mult=2): 
    dk = np.concatenate((kk[1:]-kk[:-1], [kk[-1]-kk[-2]])) # this is true for k >> kf
    Nmode = 4 * np.pi * kk**2 * dk * (Vs / (2*np.pi)**3)
    mu_arr = np.linspace(0., 1., 200)
    k_mesh, mu_mesh = np.meshgrid(kk, mu_arr, indexing='ij')
    legendre_mesh = np.array([legendre(2*l)(mu_mesh) for l in range(mult)])
    legendre_ell_mesh = np.array([(2*(2*l)+1)*legendre(2*l)(mu_mesh) for l in range(mult)])
    pkmu_mesh = (b1 + f1 * mu_mesh**2)**2 * ipklin(k_mesh)
    integrand_mu_mesh = np.einsum('k,km,lkm,pkm->lpkm', 1./Nmode, (pkmu_mesh + 1/nbar)**2,
                                  legendre_ell_mesh, legendre_ell_mesh)
    cov_diagonal = 2 * np.trapz(integrand_mu_mesh, x=mu_arr, axis=-1)
    return np.block([[np.diag(cov_diagonal[i,j]) for i in range(mult)] for j in range(mult)])


def get_growth_factor_from_params(params, parameters_dicts):
    z = params[-1]
    M = Class()
    class_dict = {}

    for i, param in enumerate(parameters_dicts):
        if param["name"] != "z":
            class_dict[param["name"]] = params[i]
            
    M.set(class_dict)

    M.compute()

    D1, f1, H, DA = M.scale_independent_growth_factor(z), M.scale_independent_growth_factor_f(z), M.Hubble(z)/M.Hubble(0.), M.angular_distance(z) * M.Hubble(0.)

    return D1, f1, H, DA


#code copied from https://github.com/DifferentiableUniverseInitiative/jax_cosmo/blob/816069f0e69d75ec83689406623839b53fbf43fc/jax_cosmo/scipy/interpolate.py#L1
# This module contains some missing ops from jax
import functools

from jax import vmap
from jax.numpy import array
from jax.numpy import concatenate
from jax.numpy import ones
from jax.numpy import zeros
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class InterpolatedUnivariateSpline_jax(object):
    def __init__(self, x, y, k=3, endpoints="not-a-knot", coefficients=None):
        """JAX implementation of kth-order spline interpolation.

        This class aims to reproduce scipy's InterpolatedUnivariateSpline
        functionality using JAX. Not all of the original class's features
        have been implemented yet, notably
        - `w`    : no weights are used in the spline fitting.
        - `bbox` : we assume the boundary to always be [x[0], x[-1]].
        - `ext`  : extrapolation is always active, i.e., `ext` = 0.
        - `k`    : orders `k` > 3 are not available.
        - `check_finite` : no such check is performed.

        (The relevant lines from the original docstring have been included
        in the following.)

        Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.
        Spline function passes through all provided points. Equivalent to
        `UnivariateSpline` with s = 0.

        Parameters
        ----------
        x : (N,) array_like
            Input dimension of data points -- must be strictly increasing
        y : (N,) array_like
            input dimension of data points
        k : int, optional
            Degree of the smoothing spline.  Must be 1 <= `k` <= 3.
        endpoints : str, optional, one of {'natural', 'not-a-knot'}
            Endpoint condition for cubic splines, i.e., `k` = 3.
            'natural' endpoints enforce a vanishing second derivative
            of the spline at the two endpoints, while 'not-a-knot'
            ensures that the third derivatives are equal for the two
            left-most `x` of the domain, as well as for the two
            right-most `x`. The original scipy implementation uses
            'not-a-knot'.
        coefficients: list, optional
            Precomputed parameters for spline interpolation. Shouldn't be set
            manually.

        See Also
        --------
        UnivariateSpline : Superclass -- allows knots to be selected by a
            smoothing condition
        LSQUnivariateSpline : spline for which knots are user-selected
        splrep : An older, non object-oriented wrapping of FITPACK
        splev, sproot, splint, spalde
        BivariateSpline : A similar class for two-dimensional spline interpolation

        Notes
        -----
        The number of data points must be larger than the spline degree `k`.

        The general form of the spline can be written as
          f[i](x) = a[i] + b[i](x - x[i]) + c[i](x - x[i])^2 + d[i](x - x[i])^3,
          i = 0, ..., n-1,
        where d = 0 for `k` = 2, and c = d = 0 for `k` = 1.

        The unknown coefficients (a, b, c, d) define a symmetric, diagonal
        linear system of equations, Az = s, where z = b for `k` = 1 and `k` = 2,
        and z = c for `k` = 3. In each case, the coefficients defining each
        spline piece can be expressed in terms of only z[i], z[i+1],
        y[i], and y[i+1]. The coefficients are solved for using
        `np.linalg.solve` when `k` = 2 and `k` = 3.

        """
        # Verify inputs
        k = int(k)
        assert k in (1, 2, 3), "Order k must be in {1, 2, 3}."
        x = jnp.atleast_1d(x)
        y = jnp.atleast_1d(y)
        assert len(x) == len(y), "Input arrays must be the same length."
        assert x.ndim == 1 and y.ndim == 1, "Input arrays must be 1D."
        n_data = len(x)

        # Difference vectors
        h = jnp.diff(x)  # x[i+1] - x[i] for i=0,...,n-1
        p = jnp.diff(y)  # y[i+1] - y[i]

        if coefficients is None:
            # Build the linear system of equations depending on k
            # (No matrix necessary for k=1)
            if k == 1:
                assert n_data > 1, "Not enough input points for linear spline."
                coefficients = p / h

            if k == 2:
                assert n_data > 2, "Not enough input points for quadratic spline."
                assert endpoints == "not-a-knot"  # I have only validated this
                # And actually I think it's probably the best choice of border condition

                # The knots are actually in between data points
                knots = (x[1:] + x[:-1]) / 2.0
                # We add 2 artificial knots before and after
                knots = jnp.concatenate(
                    [
                        jnp.array([x[0] - (x[1] - x[0]) / 2.0]),
                        knots,
                        jnp.array([x[-1] + (x[-1] - x[-2]) / 2.0]),
                    ]
                )
                n = len(knots)
                # Compute interval lenghts for these new knots
                h = jnp.diff(knots)
                # postition of data point inside the interval
                dt = x - knots[:-1]

                # Now we build the system natrix
                A = jnp.diag(
                    jnp.concatenate(
                        [
                            jnp.ones(1),
                            (
                                2 * dt[1:]
                                - dt[1:] ** 2 / h[1:]
                                - dt[:-1] ** 2 / h[:-1]
                                + h[:-1]
                            ),
                            jnp.ones(1),
                        ]
                    )
                )

                A += jnp.diag(
                    jnp.concatenate([-jnp.array([1 + h[0] / h[1]]), dt[1:] ** 2 / h[1:]]),
                    k=1,
                )
                A += jnp.diag(
                    jnp.concatenate([jnp.atleast_1d(h[0] / h[1]), jnp.zeros(n - 3)]), k=2
                )

                A += jnp.diag(
                    jnp.concatenate(
                        [
                            h[:-1] - 2 * dt[:-1] + dt[:-1] ** 2 / h[:-1],
                            -jnp.array([1 + h[-1] / h[-2]]),
                        ]
                    ),
                    k=-1,
                )
                A += jnp.diag(
                    jnp.concatenate([jnp.zeros(n - 3), jnp.atleast_1d(h[-1] / h[-2])]),
                    k=-2,
                )

                # And now we build the RHS vector
                s = jnp.concatenate([jnp.zeros(1), 2 * p, jnp.zeros(1)])

                # Compute spline coefficients by solving the system
                coefficients = jnp.linalg.solve(A, s)

            if k == 3:
                assert n_data > 3, "Not enough input points for cubic spline."
                if endpoints not in ("natural", "not-a-knot"):
                    print("Warning : endpoints not recognized. Using natural.")
                    endpoints = "natural"

                # Special values for the first and last equations
                zero = array([0.0])
                one = array([1.0])
                A00 = one if endpoints == "natural" else array([h[1]])
                A01 = zero if endpoints == "natural" else array([-(h[0] + h[1])])
                A02 = zero if endpoints == "natural" else array([h[0]])
                ANN = one if endpoints == "natural" else array([h[-2]])
                AN1 = (
                    -one if endpoints == "natural" else array([-(h[-2] + h[-1])])
                )  # A[N, N-1]
                AN2 = zero if endpoints == "natural" else array([h[-1]])  # A[N, N-2]

                # Construct the tri-diagonal matrix A
                A = jnp.diag(concatenate((A00, 2 * (h[:-1] + h[1:]), ANN)))
                upper_diag1 = jnp.diag(concatenate((A01, h[1:])), k=1)
                upper_diag2 = jnp.diag(concatenate((A02, zeros(n_data - 3))), k=2)
                lower_diag1 = jnp.diag(concatenate((h[:-1], AN1)), k=-1)
                lower_diag2 = jnp.diag(concatenate((zeros(n_data - 3), AN2)), k=-2)
                A += upper_diag1 + upper_diag2 + lower_diag1 + lower_diag2

                # Construct RHS vector s
                center = 3 * (p[1:] / h[1:] - p[:-1] / h[:-1])
                s = concatenate((zero, center, zero))
                # Compute spline coefficients by solving the system
                coefficients = jnp.linalg.solve(A, s)

        # Saving spline parameters for evaluation later
        self.k = k
        self._x = x
        self._y = y
        self._coefficients = coefficients
        self._endpoints = endpoints

    # Operations for flattening/unflattening representation
    def tree_flatten(self):
        children = (self._x, self._y, self._coefficients)
        aux_data = {"endpoints": self._endpoints, "k": self.k}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        x, y, coefficients = children
        return cls(x, y, coefficients=coefficients, **aux_data)

    def __call__(self, x):
        """Evaluation of the spline.

        Notes
        -----
        Values are extrapolated if x is outside of the original domain
        of knots. If x is less than the left-most knot, the spline piece
        f[0] is used for the evaluation; similarly for x beyond the
        right-most point.

        """
        if self.k == 1:
            t, a, b = self._compute_coeffs(x)
            result = a + b * t

        if self.k == 2:
            t, a, b, c = self._compute_coeffs(x)
            result = a + b * t + c * t**2

        if self.k == 3:
            t, a, b, c, d = self._compute_coeffs(x)
            result = a + b * t + c * t**2 + d * t**3

        return result

    def _compute_coeffs(self, xs):
        """Compute the spline coefficients for a given x."""
        # Retrieve parameters
        x, y, coefficients = self._x, self._y, self._coefficients

        # In case of quadratic, we redefine the knots
        if self.k == 2:
            knots = (x[1:] + x[:-1]) / 2.0
            # We add 2 artificial knots before and after
            knots = jnp.concatenate(
                [
                    jnp.array([x[0] - (x[1] - x[0]) / 2.0]),
                    knots,
                    jnp.array([x[-1] + (x[-1] - x[-2]) / 2.0]),
                ]
            )
        else:
            knots = x

        # Determine the interval that x lies in
        ind = jnp.digitize(xs, knots) - 1
        # Include the right endpoint in spline piece C[m-1]
        ind = jnp.clip(ind, 0, len(knots) - 2)
        t = xs - knots[ind]
        h = jnp.diff(knots)[ind]

        if self.k == 1:
            a = y[ind]
            result = (t, a, coefficients[ind])

        if self.k == 2:
            dt = (x - knots[:-1])[ind]
            b = coefficients[ind]
            b1 = coefficients[ind + 1]
            a = y[ind] - b * dt - (b1 - b) * dt**2 / (2 * h)
            c = (b1 - b) / (2 * h)
            result = (t, a, b, c)

        if self.k == 3:
            c = coefficients[ind]
            c1 = coefficients[ind + 1]
            a = y[ind]
            a1 = y[ind + 1]
            b = (a1 - a) / h - (2 * c + c1) * h / 3.0
            d = (c1 - c) / (3 * h)
            result = (t, a, b, c, d)

        return result

    def derivative(self, x, n=1):
        """Analytic nth derivative of the spline.

        The spline has derivatives up to its order k.

        """
        assert n in range(self.k + 1), "Invalid n."

        if n == 0:
            result = self.__call__(x)
        else:
            # Linear
            if self.k == 1:
                t, a, b = self._compute_coeffs(x)
                result = b

            # Quadratic
            if self.k == 2:
                t, a, b, c = self._compute_coeffs(x)
                if n == 1:
                    result = b + 2 * c * t
                if n == 2:
                    result = 2 * c

            # Cubic
            if self.k == 3:
                t, a, b, c, d = self._compute_coeffs(x)
                if n == 1:
                    result = b + 2 * c * t + 3 * d * t**2
                if n == 2:
                    result = 2 * c + 6 * d * t
                if n == 3:
                    result = 6 * d

        return result

    def antiderivative(self, xs):
        """
        Computes the antiderivative of first order of this spline
        """
        # Retrieve parameters
        x, y, coefficients = self._x, self._y, self._coefficients

        # In case of quadratic, we redefine the knots
        if self.k == 2:
            knots = (x[1:] + x[:-1]) / 2.0
            # We add 2 artificial knots before and after
            knots = jnp.concatenate(
                [
                    jnp.array([x[0] - (x[1] - x[0]) / 2.0]),
                    knots,
                    jnp.array([x[-1] + (x[-1] - x[-2]) / 2.0]),
                ]
            )
        else:
            knots = x

        # Determine the interval that x lies in
        ind = jnp.digitize(xs, knots) - 1
        # Include the right endpoint in spline piece C[m-1]
        ind = jnp.clip(ind, 0, len(knots) - 2)
        t = xs - knots[ind]

        if self.k == 1:
            a = y[:-1]
            b = coefficients
            h = jnp.diff(knots)
            cst = jnp.concatenate([jnp.zeros(1), jnp.cumsum(a * h + b * h**2 / 2)])
            return cst[ind] + a[ind] * t + b[ind] * t**2 / 2

        if self.k == 2:
            h = jnp.diff(knots)
            dt = x - knots[:-1]
            b = coefficients[:-1]
            b1 = coefficients[1:]
            a = y - b * dt - (b1 - b) * dt**2 / (2 * h)
            c = (b1 - b) / (2 * h)
            cst = jnp.concatenate(
                [jnp.zeros(1), jnp.cumsum(a * h + b * h**2 / 2 + c * h**3 / 3)]
            )
            return cst[ind] + a[ind] * t + b[ind] * t**2 / 2 + c[ind] * t**3 / 3

        if self.k == 3:
            h = jnp.diff(knots)
            c = coefficients[:-1]
            c1 = coefficients[1:]
            a = y[:-1]
            a1 = y[1:]
            b = (a1 - a) / h - (2 * c + c1) * h / 3.0
            d = (c1 - c) / (3 * h)
            cst = jnp.concatenate(
                [
                    jnp.zeros(1),
                    jnp.cumsum(a * h + b * h**2 / 2 + c * h**3 / 3 + d * h**4 / 4),
                ]
            )
            return (
                cst[ind]
                + a[ind] * t
                + b[ind] * t**2 / 2
                + c[ind] * t**3 / 3
                + d[ind] * t**4 / 4
            )

    def integral(self, a, b):
        """
        Compute a definite integral over a piecewise polynomial.
        Parameters
        ----------
        a : float
            Lower integration bound
        b : float
            Upper integration bound
        Returns
        -------
        ig : array_like
            Definite integral of the piecewise polynomial over [a, b]
        """
        # Swap integration bounds if needed
        sign = 1
        if b < a:
            a, b = b, a
            sign = -1
        xs = jnp.array([a, b])
        return sign * jnp.diff(self.antiderivative(xs))

