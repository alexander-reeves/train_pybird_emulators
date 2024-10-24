import h5py
from scipy.interpolate import interp1d, make_interp_spline
import argparse
from sklearn.decomposition import PCA
from tqdm import tqdm
import pickle
from classy import Class

from cosmic_toolbox import logger
import jax
jax.config.update("jax_enable_x64", True)
from flowMC.nfmodel.realNVP import RealNVP
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
import os
import optax
import equinox as eqx  # Equinox
import numpy as np
import jax.numpy as jnp



LOGGER = logger.get_logger(__name__)


def setup(args):
    # The setup function gets executed first before esub starts (useful to create directories for example)

    parser = argparse.ArgumentParser()

    parser.add_argument("--params_input_file", type=str, required=True)
    parser.add_argument("--ntrain", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=False, default=1000)
    parser.add_argument("--model_name", type=str, required=False, default="nf_model")
    parser.add_argument("--layer_size", type=int, required=False, default=128)
    parser.add_argument("--n_hidden", type=int, required=False, default=3)
    parser.add_argument("--num_layers", type=int, required=False, default=3)
    parser.add_argument("--batch_size", type=int, required=False, default=8192)
    parser.add_argument("--nbins", type=int, required=False, default=16)

    args = parser.parse_args(args)

    return args


def main(indices, args):
    for index in indices:
        args = setup(args)

        # Load the data
        data = np.load(args.params_input_file)[:args.ntrain]
        # Model parameters
        n_feature = data.shape[1]
        n_layers = args.num_layers
        n_hiddens = [args.layer_size] * args.n_hidden
        n_bins = args.nbins
        key, subkey = jax.random.split(jax.random.PRNGKey(1))

        model = MaskedCouplingRQSpline(
            n_feature,
            n_layers,
            n_hiddens,
            n_bins,
            subkey,
            data_cov=jnp.cov(data.T),
            data_mean=jnp.mean(data, axis=0),
        )

        num_epochs = args.epochs
        batch_size = args.batch_size
        learning_rate = 1e-3

        optim = optax.adam(learning_rate)
        state = optim.init(eqx.filter(model, eqx.is_array))
        key, subkey = jax.random.split(key)
        LOGGER.info("Training the model")
        key, model, state, loss = model.train(key, data, optim, state, num_epochs, batch_size, verbose=True)
        LOGGER.info(f"Final loss: {loss}")
        model.save_model(f"/cluster/work/refregier/alexree/local_packages/train_pybird_emulators/src/train_pybird_emulators/data/nf_models/{args.model_name}")

        yield index