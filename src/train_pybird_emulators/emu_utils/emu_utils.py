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
import h5py
from cosmic_toolbox import logger
from pybird.greenfunction import GreenFunction
import os
from train_pybird_emulators.emu_utils.interpolated_univariate_spline_jax import (
    InterpolatedUnivariateSpline_jax,
)

LOGGER = logger.get_logger(__name__)


def read_yaml_file(filename):
    with open(filename, "r") as file:
        data = yaml.safe_load(file)
    return data


def sample_from_hypercube(
    lhc, prior_ranges, dist="uniform", cov_file=None, mu_file=None
):
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
        sampled_values = mu_vector + transformed_lhc @ cholesky_decomposition.T

        return sampled_values


class PiecewiseSpline_jax:
    def __init__(self, knots, logpk_knots, dlogpk_r=1, boundaries=False):
        # dummy dlogpkr value if none supplied
        self.knots = knots
        self.logknots = jnp.log(knots)
        self.logpk_knots = logpk_knots

        # Store boundary values and their derivatives
        self.k_l = self.knots[0]
        self.k_r = self.knots[-1]

        self.nr = dlogpk_r
        self.logAr = self.logpk_knots[-1] - self.nr * self.logknots[-1]
        self.logAl = self.logpk_knots[0] - 1.0 * self.logknots[0]

        # Calculate the gradient at the first knot for left extrapolation
        # self.gradient_left = (logpk_knots[1] - logpk_knots[0]) / (self.logknots[1] - self.logknots[0])
        self.gradient_left = 1  # set this to be 1 always

        # use JAX spline
        if boundaries:
            p1_l = self.logAl + 1 * jnp.log(self.k_l * 0.999)
            p2_l = self.logAl + 1 * jnp.log(self.k_l * 0.998)
            p3_l = self.logAl + 1 * jnp.log(self.k_l * 0.997)

            p1_r = self.logAr + self.nr * jnp.log(self.k_r * 1.001)
            p2_r = self.logAr + self.nr * jnp.log(self.k_r * 1.002)
            p3_r = self.logAr + self.nr * jnp.log(self.k_r * 1.003)

            self.logknots = jnp.concatenate(
                (
                    jnp.log(
                        jnp.array(
                            [self.k_l * 0.997, self.k_l * 0.998, self.k_l * 0.999]
                        )
                    ),
                    self.logknots,
                    jnp.log(
                        jnp.array(
                            [self.k_r * 1.001, self.k_r * 1.002, self.k_r * 1.003]
                        )
                    ),
                )
            )
            self.logpk_knots = jnp.concatenate(
                (
                    jnp.array([p3_l, p2_l, p1_l]),
                    self.logpk_knots,
                    jnp.array([p1_r, p2_r, p3_r]),
                )
            )
            self.ilogpk_spline = InterpolatedUnivariateSpline_jax(
                self.logknots, self.logpk_knots, k=3, endpoints="natural"
            )

        else:
            self.ilogpk_spline = InterpolatedUnivariateSpline_jax(
                self.logknots, self.logpk_knots, k=3, endpoints="natural"
            )

    def __call__(self, k):
        # vectorize the function over k using Jax
        return jax.vmap(self.evaluate)(k)

    def evaluate(self, logk):
        k = jnp.exp(logk)
        # Use jnp.where to handle conditional logic
        return jnp.where(
            k < self.k_l,
            self.logAl + 1.0 * logk,
            jnp.where(
                k > self.k_r, self.logAr + self.nr * logk, self.ilogpk_spline(logk)
            ),
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
        self.ilogpk_spline = make_interp_spline(
            self.logknots, self.logpk_knots, k=3, bc_type=self.bc
        )

        # Store boundary values and their derivatives
        self.k_l = self.knots[0]
        self.k_r = self.knots[-1]

        self.nr = dlogpk_r
        self.logAr = self.logpk_knots[-1] - self.nr * self.logknots[-1]
        self.logAl = self.logpk_knots[0] - 1.0 * self.logknots[0]

        # Calculate the gradient at the first knot for left extrapolation
        # self.gradient_left = (logpk_knots[1] - logpk_knots[0]) / (self.logknots[1] - self.logknots[0])
        self.gradient_left = 1  # set this to be 1 always

    def __call__(self, k):
        return np.vectorize(self.evaluate)(k)

    def evaluate(self, logk):
        k = np.exp(logk)
        if k < self.k_l:
            return self.logAl + 1.0 * logk
        elif k > self.k_r:
            return self.logAr + self.nr * logk
        else:
            return self.ilogpk_spline(logk)


def get_pgg_from_linps(pk_lin, kk, eft_params, N):
    """
    Function to compute the galaxy redshift space multipoles given an inputs linear matter power spectrum
    """

    # HMM this is the bit we should ask about-- there is no "true" plin in this case
    N.compute(
        {
            "kk": kk,
            "pk_lin": pk_lin,  # pk_lin normalized by A
            "pk_lin_2": pk_lin,  # pk_lin_2 goes inside the loop integrals (+IR corr.)
            "D": 1.0,
            "f": 1.0,
            "z": 1.0,
            "Omega0_m": 1.0,
        },  # a bunch of unused dummy values
        do_core=True,
        do_survey_specific=False,
    )  # this step computes the following:

    outputs = [
        N.bird.P11l.astype(np.float32),
        N.bird.Pctl.astype(np.float32),
        N.bird.Ploopl.astype(np.float32),
        N.bird.IRPs11.astype(np.float32),
        N.bird.IRPsct.astype(np.float32),
        N.bird.IRPsloop.astype(np.float32),
    ]

    return outputs


def get_pgg_from_linps_and_f_and_A(pk_lin, kk, N, f, A):
    """
    Function to compute the galaxy redshift space multipoles given an inputs linear matter power spectrum
    """

    N.compute(
        {
            "kk": kk,
            "pk_lin": pk_lin,  # pk_lin normalized by A
            "pk_lin_2": pk_lin,  # pk_lin_2 goes inside the loop integrals (+IR corr.)
            "D": 1.0,
            "f": 1.0,
            "z": 1.0,
            "Omega0_m": 1.0,
        },  # a bunch of unused dummy values
        do_core=True,
        do_survey_specific=False,
    )  # this step computes the following:

    # Do the set resum part
    Q = N.resum.makeQ(f)
    Dp2 = A
    Dp2n = np.concatenate(
        (2 * [N.co.Na * [Dp2 ** (n + 1)] for n in range(N.co.NIR)])
    )  # function from PyBird but

    N.bird.IRPs11 = np.einsum("n,lnk->lnk", Dp2n, N.bird.IRPs11)
    N.bird.IRPsct = np.einsum("n,lnk->lnk", Dp2n, N.bird.IRPsct)
    N.bird.IRPsloop = np.einsum("n,lmnk->lmnk", Dp2n, N.bird.IRPsloop)

    N.bird.setIRPs(Q=Q)

    # Still keep the pieces separate as we might not want to resum
    outputs = [
        N.bird.P11l.astype(np.float32),
        N.bird.Pctl.astype(np.float32),
        N.bird.Ploopl.astype(np.float32),
        N.bird.fullIRPs11.astype(np.float32),
        N.bird.fullIRPsct.astype(np.float32),
        N.bird.fullIRPsloop.astype(np.float32),
    ]

    return outputs


def get_pgg_from_params(params, parameters_dicts, N, kk, knots):

    input_dict = {}
    for i, param in enumerate(parameters_dicts):
        input_dict[param["name"]] = params[i]

    pkmax = input_dict["pk_max"]
    f = input_dict["f"]
    # get the logpk_knots
    logpk_knots = [
        input_dict[key]
        for key in sorted(
            input_dict.keys(),
            key=lambda x: int(x.split("_")[-1])
            if x.startswith("pk_knot_")
            else float("inf"),
        )
        if key.startswith("pk_knot_")
    ]
    logpk_knots = np.array(logpk_knots)

    # Make the spline for the linear MPS
    ipk_loglog_spline = PiecewiseSpline_jax(knots, logpk_knots)
    pk_lin = np.exp(ipk_loglog_spline(np.log(kk)))  # In Mpc/h units everywhere

    P11l, Pctl, Ploopl, IRPs11, IRPsct, IRPsloop = get_pgg_from_linps_and_f_and_A(
        pk_lin, kk, N, f, pkmax
    )

    return P11l, Pctl, Ploopl, IRPs11, IRPsct, IRPsloop


def to_Mpc_per_h(_pk, _kk, h):
    ilogpk_ = interp1d(np.log(_kk), np.log(_pk), fill_value="extrapolate")
    return np.exp(ilogpk_(np.log(_kk * h))) * h**3


def get_logslope(x, f, side="left"):
    if side == "left":
        n = (np.log(f[1]) - np.log(f[0])) / (np.log(x[1]) - np.log(x[0]))
        A = f[0] / x[0] ** n
    elif side == "right":
        n = (np.log(f[-1]) - np.log(f[-2])) / (np.log(x[-1]) - np.log(x[-2]))
        A = f[-1] / x[-1] ** n
    return A, n


def compute_1loop_bpk(pk, f, D, z, kk, eft_params, knots, k_l, k_r, resum=False):
    """
    Function to compute the original pybird bpk prediction for later comparison with the emulator performance
    """
    # Maybe keep this outside function for speed
    N = Correlator()
    N.set(
        {
            "output": "bPk",
            "multipole": 3,
            "kmax": 0.4,
            "fftaccboost": 2,  # boosting the FFTLog precision (slower, but ~0.1% more precise -> let's emulate this)
            "with_resum": resum,
            "with_exact_time": True,
            "km": 1.0,
            "kr": 1.0,
            "nd": 3e-4,
            "eft_basis": "eftoflss",
            "with_stoch": True,
        }
    )

    N.compute(
        {"kk": kk, "pk_lin": pk, "D": D, "f": f, "z": z, "Omega0_m": Omega0_m},
        do_core=True,
        do_survey_specific=True,
    )

    bpk_true = N.get(eft_params)

    return bpk_true


def test_1loop_sparsity(cosmo_dict, kk, eft_params):
    """
    Function to compute the difference between the reconstructed and original 1-loop matter power given
    an input cosmology dictionary and optimal knots
    """
    # hardcoded for now
    k_l, k_r = 1e-4, 1.0  # In Mpc/h
    # Maybe keep this outside function for speed this sets up for real-space mps
    N = Correlator()

    N.set(
        {
            "output": "bPk",
            "multipole": 3,
            "kmax": 0.6,
            "fftaccboost": 2,  # boosting the FFTLog precision (slower, but ~0.1% more precise -> let's emulate this)
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

    # First use class to compute the linear growth factor and the f growth factor
    M = Class()
    keys_to_exclude = ["z", "pk_max"]
    cosmo = {k: v for k, v in cosmo_dict.items() if k not in keys_to_exclude}
    M.set(cosmo)
    M.set({"output": "mPk", "P_k_max_h/Mpc": 5, "z_max_pk": cosmo_dict["z"]})
    M.compute()
    pk_class = np.array(
        [M.pk_lin(k * M.h(), cosmo_dict["z"]) * M.h() ** 3 for k in kk]
    )  # k in Mpc/h, pk in (Mpc/h)^3

    # Linear growth factor and other pieces required
    A_s, Omega0_m = 1e-10 * np.exp(cosmo_dict["ln10^{10}A_s"]), M.Omega0_m()
    D1, f1 = (
        M.scale_independent_growth_factor(cosmo_dict["z"]),
        M.scale_independent_growth_factor_f(cosmo_dict["z"]),
    )

    N.compute(
        {
            "kk": kk,
            "pk_lin": pk_class,
            "pk_lin_2": pk_class,
            "D": D1,
            "f": f1,
            "z": cosmo_dict["z"],
            "Omega0_m": Omega0_m,
        },
        do_core=True,
        do_survey_specific=True,
    )

    bpk_0 = N.get(eft_params)  # original bpk to compare size with

    # Now loop through and find out the indices of small values of the IRloops
    Q = 1.0 * N.bird.Q
    M0 = 1.0 * N.bird.IRPsloop

    for p in range(M0.shape[0]):
        for i in range(M0.shape[1]):
            for n in range(M0.shape[2]):
                if (
                    Q[1, :, p, n] == 0
                ).all():  # if they are multiplied by 0, setting them to 0
                    M0[p, i, n] = 0.0

    sparsity = 1 - np.count_nonzero(M0) / M0.size
    print("sparsity")
    print("%.3f" % sparsity)

    C = csr_matrix(
        M0.reshape(-1)
    )  # this is the sparse array stored in a compressed format
    # print (C.shape)
    C_idx = C.indices  # non-zero indices

    return C_idx


def get_default_cov():
    z = 0.5
    kk = np.logspace(-5, 0, 1000)
    M = Class()
    cosmo = {
        "omega_b": 0.02235,
        "omega_cdm": 0.120,
        "h": 0.675,
        "ln10^{10}A_s": 3.044,
        "n_s": 0.965,
    }
    M.set(cosmo)
    M.set({"output": "mPk", "P_k_max_h/Mpc": 1, "z_max_pk": z})
    M.compute()

    ## [Mpc/h]^3
    Vs = 1.0e11  # ideal volume (in [Mpc/h]^3), > 3 x better than DESI / Euclid (BOSS ~ 3e9, DESI ~ 3e10)
    nbar = 5e-3  # ideal nbar (for b1~2) (in [Mpc/h]^3), > 3 x better than DESI / Euclid

    pk_lin = np.array(
        [M.pk_lin(k * M.h(), z) * M.h() ** 3 for k in kk]
    )  # k in Mpc/h, pk in (Mpc/h)^3
    ipk_h = interp1d(kk, pk_lin, kind="cubic")
    cov = get_cov(kk, ipk_h, 1, 0.0, Vs, nbar=nbar, mult=3)

    return cov


def get_cov(kk, ipklin, b1, f1, Vs, nbar=3.0e-4, mult=2):
    dk = np.concatenate(
        (kk[1:] - kk[:-1], [kk[-1] - kk[-2]])
    )  # this is true for k >> kf
    Nmode = 4 * np.pi * kk**2 * dk * (Vs / (2 * np.pi) ** 3)
    mu_arr = np.linspace(0.0, 1.0, 200)
    k_mesh, mu_mesh = np.meshgrid(kk, mu_arr, indexing="ij")
    legendre_mesh = np.array([legendre(2 * l)(mu_mesh) for l in range(mult)])
    legendre_ell_mesh = np.array(
        [(2 * (2 * l) + 1) * legendre(2 * l)(mu_mesh) for l in range(mult)]
    )
    pkmu_mesh = (b1 + f1 * mu_mesh**2) ** 2 * ipklin(k_mesh)
    integrand_mu_mesh = np.einsum(
        "k,km,lkm,pkm->lpkm",
        1.0 / Nmode,
        (pkmu_mesh + 1 / nbar) ** 2,
        legendre_ell_mesh,
        legendre_ell_mesh,
    )
    cov_diagonal = 2 * np.trapz(integrand_mu_mesh, x=mu_arr, axis=-1)
    return np.block(
        [[np.diag(cov_diagonal[i, j]) for i in range(mult)] for j in range(mult)]
    )


def get_growth_factor_from_params(params, parameters_dicts):
    z = params[-1]
    M = Class()
    class_dict = {}

    for i, param in enumerate(parameters_dicts):
        if param["name"] != "z":
            class_dict[param["name"]] = params[i]

    M.set(class_dict)

    M.compute()

    D1, f1, H, DA = (
        M.scale_independent_growth_factor(z),
        M.scale_independent_growth_factor_f(z),
        M.Hubble(z) / M.Hubble(0.0),
        M.angular_distance(z) * M.Hubble(0.0),
    )

    return D1, f1, H, DA


def get_green_function_from_params(params, parameters_dicts):

    input_dict = {}
    for i, param in enumerate(parameters_dicts):
        input_dict[param["name"]] = params[i]

    GF = GreenFunction(
        Omega0_m=input_dict["omega_m"], w=input_dict["w0"], quintessence=True
    )
    a = 1 / (1 + input_dict["z"])
    Y1 = GF.Y(a)
    G1t = GF.mG1t(a)
    V12t = GF.mV12t(a)
    G1 = GF.G(a)
    f = GF.fplus(a)

    return Y1, G1t, V12t, G1, f


# Function to load and save data to HDF5
def update_or_create_dataset(dataset_name, data, hdf_file):
    if dataset_name in hdf_file:
        # Dataset exists, append the data to the existing dataset
        hdf_file[dataset_name].resize(
            (hdf_file[dataset_name].shape[0] + data.shape[0]), axis=0
        )
        hdf_file[dataset_name][-data.shape[0] :] = data
    else:
        # Dataset doesn't exist, create it
        if dataset_name in ["params", "pk_lin"]:
            maxshape = (None, data.shape[1])
        else:
            maxshape = (None,)
        hdf_file.create_dataset(dataset_name, data=data, maxshape=maxshape)


def generate_repeating_array(original_array, segment_length, n):
    result_array = []

    for i in range(0, len(original_array), segment_length):
        segment = original_array[i : i + segment_length]
        repeated_segment = np.tile(segment, n)

        result_array.extend(repeated_segment)

    return result_array


def rel2realpath(rel_path):
    """
    Simple helper function to change the relative path to a real path on user's computer.

    :param rel_path: the relative path to a file in a directory

    :return: a real full-path to the file to be read in
    """
    path2script = os.path.dirname(os.path.abspath(__file__))

    real_path = os.path.realpath(os.path.join(path2script, rel_path))

    return real_path


# remove Nans
def remove_nan_rows_from_both_arrays(y_train, x_train):
    """
    Remove rows from data_array that contain any NaN values and remove corresponding rows from x_train.
    Parameters:
    - data_array: 2D numpy array of shape (200000, 10185)
    - x_train: 2D numpy array of shape (200000, 47)
    Returns:
    - Cleaned data_array and x_train with rows containing NaN values removed.
    """
    nan_row_indices = np.where(np.isnan(y_train).any(axis=1))[0]

    cleaned_data_array = np.delete(y_train, nan_row_indices, axis=0)
    cleaned_x_train = np.delete(x_train, nan_row_indices, axis=0)

    return cleaned_data_array, cleaned_x_train


def get_training_data_from_hdf5(fn, piece_name, ntrain, mono, quad_hex):

    with h5py.File(fn, "r") as f:
        LOGGER.info(
            f"total number of available training points: {f['params'].shape[0]}"
        )
        LOGGER.info(f"Available keys in the file: {f.keys()}")
        x_train = f["params"][:ntrain]

        if piece_name is not None:
            if mono:
                LOGGER.info(f"Using monopole data for {piece_name}")
                y_train = f[f"{piece_name}"][:ntrain, : 35 * 97]

            if quad_hex:
                LOGGER.info(f"Using quadhex data for {piece_name}")
                y_train = f[f"{piece_name}"][:ntrain, 35 * 97 :]

            if not mono and not quad_hex:
                y_train = f[f"{piece_name}"][:ntrain]

        else:
            LOGGER.info("No piece name provided, using all columns of the data")
            y_train = np.vstack(
                [f[key][:ntrain] for key in f.keys() if key != "params"]
            ).T

        return x_train, y_train
