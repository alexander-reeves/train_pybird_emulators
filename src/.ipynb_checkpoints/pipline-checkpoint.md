# The pipline


- copy the files in the lhc_bank_z0p5to2p5 and put it under train_pybird_emulators/lhc_bank_z0p5to2p5/
  The data can be found at the github FreqCosmo

- generate knotes: esub knots_decomposition.py --nknots 55
  There some modifications to the knots_decomposition.py
  1. In the function load_pk_cosmo_dict, lhc_pk_lin.append(to_Mpc_per_h(f[keys[i]][:], kk, lhc_cosmo_dicts[i]['h'])) is used instead of origial.
  2. In the setup, kk = np.linspace(k_l, k_r, 200) instead of 500 to match the lhc_bank..data
  3. L226knots_decomposition         nlog = 45,nlog_lowk = 5 to allow nknots=58

- run train: esub generate_training_data.py --num_samps_per_index 1 --filename_config ../submission_scripts/config_growth.yaml --filename_knots optimised_knots_cov_err/knots_temp_file_knots_58_index_0.npy --n_gridpoints 58 --function='merge'
  Some dangerous modifications
  1. L146generate_training_data: logpk_knots, pkmax, f = np.array(params), params[-2], params[-1]
     where originally logpk_knots=np.array(params[:-2]) and thus has shape 56,
  2. L51emu_utils: sampled_values = mu_vector + transformed_lhc.T @ cholesky_decomposition
 
- train:
  esub train_model_gpu_growth.py --training_data_file pybird_emu_training_data_80_knots/total_data.h5 --ntrain 2 --epochs 2
  esub train_model_gpu.py --training_data_file pybird_emu_training_data_80_knots/total_data.h5 --ntrain 2 --epochs 2 --piece_name P11l