from pybird import config

config.set_jax_enabled(True)  # Enable JAX by setting the config Class
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from copy import copy, deepcopy
from scipy.interpolate import interp1d, make_interp_spline
import h5py
import time
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
import utils
import jax
from jax.config import config

config.update("jax_enable_x64", True)
# from jax.scipy import optimize
import jax.numpy as jnp
from jax import vmap
import logging
from scipy import optimize

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def load_pk(file_path, nbank, kk):
    with h5py.File(file_path + "total_data.h5", "r") as f:
        lhc_pk_lin = f["pk_lin"][:]
        param_array = f["params"][:]

    lhc_pk_lin = jnp.array(lhc_pk_lin)

    param_order = [
        "omega_cdm",
        "omega_b",
        "h",
        "Omega_k",
        "n_s",
        "N_ur",
        "m_ncdm",
        "w0_fld",
        "z",
    ]

    lhc_cosmo_dict = {
        param_order[i]: param_array[:, i] for i in range(param_array.shape[1])
    }

    return lhc_cosmo_dict, lhc_pk_lin


def get_logslope(x, f, side="left"):
    if side == "left":
        n = (jnp.log(f[1]) - jnp.log(f[0])) / (jnp.log(x[1]) - jnp.log(x[0]))
        A = f[0] / x[0] ** n
    elif side == "right":
        n = (jnp.log(f[-1]) - jnp.log(f[-2])) / (jnp.log(x[-1]) - jnp.log(x[-2]))
        A = f[-1] / x[-1] ** n
    return A, n


def get_pybird_chi2(knots, kk, cov_inv, pk, dlogpk_r, pk_max, N, N_rec):

    eft_params = {
        "b1": 1.9536,
        "c2": 0.5865,
        "c4": 0.0,
        "b3": -0.3595,
        "cct": 0.1821,
        "cr1": -0.8417,
        "cr2": -0.808,
        "ce0": 1.4954,
        "ce1": 0.0,
        "ce2": -1.6292,
        "b2": 0.4147,
        "b4": 0.4147,
    }
    # Construct the spline
    ilogpk = utils.InterpolatedUnivariateSpline_jax(
        jnp.log(kk), jnp.log(pk), endpoints="natural"
    )
    logpk = ilogpk(jnp.log(knots))

    D1 = 0.7696605714261694
    f1 = 0.7588944657919923
    Omega0_m = 0.3
    z = 0.5

    spline = utils.PiecewiseSpline_jax(knots, logpk, dlogpk_r)

    prediction_pklin = jnp.exp(spline(jnp.log(kk)))

    einsum_str = "b,lbx->lx"

    N.compute(
        {
            "kk": kk,
            "pk_lin": jnp.array(pk),
            "D": D1,
            "f": f1,
            "z": z,
            "Omega0_m": Omega0_m,
        },
        do_core=True,
        do_survey_specific=True,
    )

    bpk_truth = N.get(eft_params) * pk_max

    N_rec.compute(
        {
            "kk": kk,
            "pk_lin": jnp.array(prediction_pklin),
            "D": D1,
            "f": f1,
            "z": z,
            "Omega0_m": Omega0_m,
        },
        do_core=True,
        do_survey_specific=True,
    )

    N_rec.get(eft_params) * pk_max

    N.bird.Ploopl = N_rec.bird.Ploopl
    N.bird.setreducePslb(N.bias)

    bpk_reconstructed = N.bird.fullPs * pk_max

    difference = bpk_truth.flatten() - bpk_reconstructed.flatten()
    chi2 = jnp.dot(difference, jnp.dot(cov_inv, difference))
    return chi2


def minimize_spline_all(knots_start, pk_info_dict_list, kk, k_l, k_r, cov_inv):
    nbank = len(pk_info_dict_list)

    # Extract fixed knots
    leftmost_knot = knots_start[0]
    rightmost_knot = knots_start[-1]

    # Internal knots to be optimized
    internal_knots_start = knots_start[1:-1]

    # Extract arrays for vectorized operations
    pk_array = jnp.array(pk_info_dict_list["pk"], dtype=jnp.float32)
    dlogpk_r_array = jnp.array(pk_info_dict_list["dlogpk_r"], dtype=jnp.float32)
    pk_max_array = jnp.array(pk_info_dict_list["pk_max"], dtype=jnp.float32)

    N = Correlator()
    N.set(
        {
            "output": "bPk",
            "multipole": 3,
            "kmax": 0.4,
            "fftaccboost": 2,
            "with_time": False,
            "with_resum": False,
            "with_exact_time": True,
            "km": 1.0,
            "kr": 1.0,
            "nd": 3e-4,
            "eft_basis": "eftoflss",
            "with_stoch": True,
        }
    )

    N_rec = Correlator()
    N_rec.set(
        {
            "output": "bPk",
            "multipole": 3,
            "kmax": 0.4,
            "fftaccboost": 2,
            "with_time": False,
            "with_resum": False,
            "with_exact_time": True,
            "km": 1.0,
            "kr": 1.0,
            "nd": 3e-4,
            "eft_basis": "eftoflss",
            "with_stoch": True,
        }
    )

    def get_chi2_vectorized(internal_knots):
        knots = jnp.concatenate(
            (jnp.array([leftmost_knot]), internal_knots, jnp.array([rightmost_knot]))
        )
        # Use jnp.diff and jnp.any within jnp.where to conditionally return jnp.inf
        condition = jnp.any(jnp.diff(knots) <= 0)
        # set up a PyBird instance to get the chi2
        def true_fn(_):
            return 1e3

        def false_fn(_):
            nbins = 77
            logger.info("done one iteration")
            return jnp.sqrt(jnp.sum(chi2) / nbank / nbins)

    # Optimization
    method = "Nelder-Mead"
    nk = len(internal_knots_start)

    get_chi2_vectorized_jit = jax.jit(get_chi2_vectorized)

    logger.info(f"Starting optimization with {nk} knots")

    # use scipy optimization
    result = optimize.minimize(
        get_chi2_vectorized_jit,
        internal_knots_start,
        method=method,
        options={"disp": True},
    )
    optimized_knots = np.concatenate(([leftmost_knot], result.x, [rightmost_knot]))

    return optimized_knots, result.fun


def setup(args):

    parser = argparse.ArgumentParser(description="Run knots optimization")
    parser.add_argument("--nknots", type=int, default=100)
    parser.add_argument("--nbank", type=int, default=100)
    parser.add_argument("--n_k", type=int, default=100)

    args = parser.parse_args(args)

    nknots = args.nknots
    nbank = args.nbank

    # The internal knots used by pybird up to kmax=0.4 which is where the emus are trained
    k_arr = np.array(
        [
            0.001,
            0.0025,
            0.005,
            0.0075,
            0.01,
            0.0125,
            0.015,
            0.0175,
            0.02,
            0.0225,
            0.025,
            0.0275,
            0.03,
            0.035,
            0.04,
            0.045,
            0.05,
            0.055,
            0.06,
            0.065,
            0.07,
            0.075,
            0.08,
            0.085,
            0.09,
            0.095,
            0.1,
            0.105,
            0.11,
            0.115,
            0.12,
            0.125,
            0.13,
            0.135,
            0.14,
            0.145,
            0.15,
            0.155,
            0.16,
            0.165,
            0.17,
            0.175,
            0.18,
            0.185,
            0.19,
            0.195,
            0.2,
            0.205,
            0.21,
            0.215,
            0.22,
            0.225,
            0.23,
            0.235,
            0.24,
            0.245,
            0.25,
            0.255,
            0.26,
            0.265,
            0.27,
            0.275,
            0.28,
            0.285,
            0.29,
            0.295,
            0.3,
            0.31,
            0.32,
            0.33,
            0.34,
            0.35,
            0.36,
            0.37,
            0.38,
            0.39,
            0.4,
        ]
    )

    lhc_cosmo_dict, lhc_pk_lin = load_pk(
        "/cluster/work/refregier/alexree/frequentist_framework/FreqCosmo/lhc_bank_z0p5to2p5/",
        nbank,
        kk,
    )

    logger.info("Loaded pk and cosmo dict")

    return nknots, k_l, k_r, kk, lhc_cosmo_dict, lhc_pk_lin, cov_inv, nbank


def main(indices, args):
    logger.info("Starting main function")
    nknots, k_l, k_r, kk, lhc_cosmo_dict, lhc_pk_lin, cov_inv, nbank = setup(args)
    Vs = 1.0e11  # ideal volume (in [Mpc/h]^3), > 3 x better than DESI / Euclid (BOSS ~ 3e9, DESI ~ 3e10)
    nbar = 6e-3  # ideal nbar (for b1~2) (in [Mpc/h]^3), > 3 x better than DESI / Euclid

    # as a set-up get dlogpk_r for each pk in the bank
    start = time.time()
    pk_info_dict_list = []

    mask = np.where((kk > k_l) & (kk < k_r))
    kk_ = kk[mask]

    logger.info("defining return info function")

    def return_info(lhc_pk_lin_i, kk, kk_):
        pk_i = 1.0 * lhc_pk_lin_i
        pk_max = jnp.max(pk_i)
        pk = 1.0 * pk_i / pk_max  # normalizing
        ilogpk = utils.InterpolatedUnivariateSpline_jax(
            jnp.log(kk), jnp.log(pk), k=3, endpoints="natural"
        )
        pkk_ = jnp.exp(ilogpk(jnp.log(kk_)))

        A_l, n_l = get_logslope(kk_, pkk_, side="left")
        A_r, n_r = get_logslope(kk_, pkk_, side="right")

        pk_r = A_r * k_r**n_r
        dpk_r = A_r * n_r * k_r ** (n_r - 1.0)
        dlogpk_r = k_r / pk_r * dpk_r

        return {"pk": pk, "dlogpk_r": dlogpk_r, "pk_max": pk_max}

    logger.info("Jitting return info")
    return_info_jit = jax.jit(return_info)

    start = time.time()

    logger.info("Starting vmap")
    # use vmap to get the info for each pk in the bank
    pk_info_dict_list = jax.vmap(return_info_jit, in_axes=(0, None, None))(
        lhc_pk_lin, kk, kk_
    )
    logger.info("Finished vmap")
    logger.info(f"Time taken to get pk_info_dict_list: {time.time()-start}")

    for index in indices:
        random_seed = index

        k_mid = 0.05
        n_low = 15
        n_high = nknots - n_low
        knots_low = np.geomspace(k_l, k_mid, n_low)
        knots_high = np.geomspace(k_mid + 0.001, k_r, n_high)
        knots = np.concatenate((knots_low, knots_high))

        # add random shifts for MonteCarlo approaxh
        def scale_knots_except_bounds(knots, scale_percentage=5):
            # Convert percentage to scale factors
            min_scale = 1 - scale_percentage / 100.0
            max_scale = 1 + scale_percentage / 100.0

            # Generate random scale factors for each knot, except the first and last
            scale_factors = np.random.uniform(min_scale, max_scale, size=knots.size - 2)

            # Scale the knots, except the first and last
            scaled_knots = np.copy(
                knots
            )  # Make a copy to preserve the original knots array
            scaled_knots[1:-1] = knots[1:-1] * scale_factors

            # Manually set the first and last knots to ensure they remain unchanged
            scaled_knots[0] = k_l
            scaled_knots[-1] = k_r

            return scaled_knots

        # Experiment with different scaling percentages
        scale_percentage = 1  # Adjust this to explore different extents of variability
        knots_start = scale_knots_except_bounds(
            knots, scale_percentage=scale_percentage
        )

        print("knots start", knots_start)
        print("diff knots start", np.diff(knots_start))
        # Optimize the knots

        knots = jnp.array(knots_start)
        minchi2, delta_chi2 = 1e16, 1e16
        t0 = time.time()
        i = 0

        threshold = 1e-6  # 1e-5
        while delta_chi2 > threshold:
            knots_i, minchi2_i = minimize_spline_all(
                knots, pk_info_dict_list, kk, k_l, k_r, cov_inv
            )
            ti = time.time()
            print(
                "step %s in %.1f sec, [red min chi2]^0.5=%.10f"
                % (i, ti - t0, minchi2_i**0.5)
            )
            delta_chi2 = abs(minchi2 - minchi2_i)
            minchi2 = 1.0 * minchi2_i
            t0 = 1.0 * ti
            i += 1

        np.save(
            f"./optimised_knots_lss_err/knots_temp_file_knots_{nknots}_index_{index}.npy",
            knots,
        )

        yield index
