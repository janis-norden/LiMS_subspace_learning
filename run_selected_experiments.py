import Experiments.experiments as experiments
import numpy as np
import time

'''

This file may be used to run any of the experiments contained in 'expeirments.py' in a specific order and with the experimental configuration 
specified in this file.
To do so, uncomment the corresponding section below and execute the code in this file.

'''

# -----------------------------------------------------------------------------------------------------------------------------------
# add time delay to start experiment
delay = 0
print('Experiment Started:')
print('Delay: ' + str(delay) + ' seconds')
time.sleep(delay)
# -----------------------------------------------------------------------------------------------------------------------------------
# set options and run experiment 1
exp1_opts = {
    'num_rand_init':            20,                                             # positive int, number of random initializations per subspace learning
    'cols_vec':                 np.array([1, 2]),                               # np array int, indicates the dimensions of subspaces to search
    'file_name_spherical':      'prednisone_3D_spherical_finite_mixture.pckl',  # name of file in data/interim/ to be loaded as density est. with spherical cov. matrices
    'file_name_full':           'prednisone_3D_full_finite_mixture.pckl'        # name of file in data/interim/ to be loaded as density est. with full cov. matrices
}
experiments.run_exp1_pred_covariances_types(exp1_opts)
# -------------------------------------------------------------------------------------------------------------------------------------
# set options and run experiment 1 (restricted number of components)
exp1_opts = {
    'num_rand_init':            20,                                                         # positive int, number of random initializations per subspace learning
    'cols_vec':                 np.array([1, 2]),                                           # np array int, indicates the dimensions of subspaces to search
    'file_name_spherical':      'prednisone_3D_spherical_finite_mixture_5_components.pckl', # name of file in data/interim/ to be loaded as density est. with spherical cov. matrices
    'file_name_full':           'prednisone_3D_full_finite_mixture_5_components.pckl'       # name of file in data/interim/ to be loaded as density est. with full cov. matrices
}
experiments.run_exp1_pred_covariances_types(exp1_opts)
# -------------------------------------------------------------------------------------------------------------------------------------
# set options and run experiment 2
exp2_opts = {
    'num_rand_init':            20,                                           # positive int, number of random initializations per subspace learning
    'cols_vec':                 np.array([1, 2]),                             # np array int, indicates the dimensions of subspaces to search
    'file_name_spherical':      'prednisone_3D_spherical_finite_mixture.pckl' # name of file in data/interim/ to be loaded as density est. with spherical cov. matrices
}
experiments.run_exp2_pred_batch_iterative(exp2_opts)
# -------------------------------------------------------------------------------------------------------------------------------------
# set options and run experiment 3
exp3_opts = {
    'num_rand_init':            20,                                           # positive int, number of random initializations per subspace learning
    'cols_vec':                 np.array([1, 2]),                             # np array int, indicates the dimensions of subspaces to search
    'file_name_spherical':      'prednisone_3D_spherical_finite_mixture.pckl' # name of file in data/interim/ to be loaded as density est. with spherical cov. matrices
}
experiments.run_exp3_pred_ssl_GLVQ(exp3_opts)
# ------------------------------------------------------------------------------------------------------------------------------------
# set options and run experiment 4
exp4_opts = {
    'num_folds':                10,                                  # positive int, number of folds for k-fold cross-validation
    'num_rand_init':            3,                                   # positive int, number of random initializations per subspace learning
    'cols_vec':                 np.array([1, 2, 3]),                 # np array int, indicates the dimensions of subspaces to search
    'file_name_spherical':      'gravitational_waves_spherical.pckl' # name of file in data/interim/ to be loaded as density est. with spherical cov. matrices
}
experiments.run_exp4_gw_spherical(exp4_opts)
# -------------------------------------------------------------------------------------------------------------------------------------
# set options and run experiment 4b
exp4b_opts = {
    'num_splits':                10,                                                 # positive int, number of folds for k-fold cross-validation
    'num_rand_init':              3,                                                 # positive int, number of random initializations per subspace learning
    'test_size':                0.2,                                                 # postive float, number between 0 and 1
    'cols_vec':                 np.array([1, 2, 3]),                                 # np array int, indicates the dimensions of subspaces to search
    'file_name_spherical':      'gravitational_waves_spherical_finite_mixture.pckl'  # name of file in data/interim/ to be loaded as density est. with spherical cov. matrices
}
experiments.run_exp4b_gw_spherical_random_subsampling(exp4b_opts)
# ------------------------------------------------------------------------------------------------------------------------------------
# set options and run experiment 5
exp5_opts = {
    'num_orth_init':             50,                                          # positive int, number of random orthogonal initializationis for subspace learning
    'file_name_spherical':      'prednisone_3D_spherical_finite_mixture.pckl' # name of file in data/interim/ to be loaded as density est. with spherical cov. matrices
}
experiments.run_exp5_pred_worst_case_inits(exp5_opts)
# ------------------------------------------------------------------------------------------------------------------------------------
# set options and run experiment 6
exp6_opts = {
    'num_rand_init':             50,                                          # positive int, number of random initializationis for subspace learning
    'file_name_spherical':      'prednisone_3D_spherical_finite_mixture.pckl' # name of file in data/interim/ to be loaded as density est. with spherical cov. matrices
}
experiments.run_exp6_pred_GMLVQ_inits(exp6_opts)
# -------------------------------------------------------------------------------------------------------------------------------------