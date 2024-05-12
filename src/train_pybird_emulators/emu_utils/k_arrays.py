
import numpy as np
from pybird.correlator import Correlator

k_emu = np.array([
    0.001, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175,
    0.02, 0.0225, 0.025, 0.0275, 0.03, 0.035, 0.04, 0.045,
    0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085,
    0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12, 0.125,
    0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165,
    0.17, 0.175, 0.18, 0.185, 0.19, 0.195, 0.2, 0.205,
    0.21, 0.215, 0.22, 0.225, 0.23, 0.235, 0.24, 0.245,
    0.25, 0.255, 0.26, 0.265, 0.27, 0.275, 0.28, 0.285,
    0.29, 0.295, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35,
    0.36, 0.37, 0.38, 0.39, 0.4
])

N = Correlator()
#Set up pybird in time unspecified mode for the computation of the pybird pieces training data
N.set({'output': 'bPk', 'multipole': 3, 'kmax': 0.6,
'fftaccboost': 2,
'with_resum': True, 'with_exact_time': True,
'with_time': False, # time unspecified
'km': 1., 'kr': 1., 'nd': 3e-4,
'eft_basis': 'eftoflss', 'with_stoch': True})

#internal pybird k-array
k_pybird = N.co.k