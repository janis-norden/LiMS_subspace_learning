import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import logging
from pathlib import Path
from copy import deepcopy

# find path to project root
path_to_project_root = str(Path(__file__).parents[1])
path_to_logs = path_to_project_root + '/src/logs/'

# configure logger
logging.basicConfig(filename = path_to_logs + 'density_estimator.log',
                    level = logging.INFO,
                    format = '%(asctime)s %(levelname)s %(message)s')

class DensityEstimator:

    ### attributes ###
    data_box = None
    density_est_opts = None

    ### methods ###
    def __init__(self, data_box, density_est_opts):
        self.data_box = deepcopy(data_box)
        self.density_est_opts = deepcopy(density_est_opts)


    def run_density_estimation(self):
        # execute density estimation with configurations specified in density_est_opts
        
        # call to density estimation
        logging.info('--- Call to density estimator ---')

        # unpack from self
        density_est_opts = self.density_est_opts

        # check density estimation type wanted
        if density_est_opts['type'] == 'bayesian_gaussian_mixture':
            data_box = self._run_bayesian_gaussian_mixture()

         # store density_est_opts in data_box
        data_box._density_est_opts = density_est_opts
            
        return data_box
    
    def _run_bayesian_gaussian_mixture(self):
        # runs Bayesian Gaussian mixture by calling sklearn

        # extract from self
        data_box_new = self.data_box
        density_est_opts = self.density_est_opts

        # extract posterior samples from data box
        posterior_samples = data_box_new.get_posterior_samples()

        # initialize dict
        density_est_dict = {}

        # loop over examples
        for i in range(0, len(data_box_new._labels)):
            
            # log iteration
            #logging.info('Bayesian Gaussian Mixture: ' + str(i + 1) + ' / ' + str(len(data_box_new._labels)))
            print('Bayesian Gaussian Mixture: ' + str(i + 1) + ' / ' + str(len(data_box_new._labels)))

            # density estimation with Bayesian Gaussian Mixture
            BGM = BayesianGaussianMixture(
                                            n_components = density_est_opts['n_components'], 
                                            weight_concentration_prior_type = 'dirichlet_distribution',
                                            n_init = density_est_opts['n_init'], 
                                            max_iter = density_est_opts['max_iter'],  
                                            covariance_type = density_est_opts['covariance_type'],
                                            random_state = 0
                                            ).fit(posterior_samples[i]['samples'])

            # sort weights in descending order
            idx_sort_desc = np.flip(np.argsort(BGM.weights_))

            # accumulate weights to 99% and find critical index
            cum_weights = np.cumsum(BGM.weights_[idx_sort_desc])
            idx_trim = np.argmax(cum_weights > density_est_opts['trim_percent']) + 1

            # trim away components that do not make the cutoff
            indices_trim = idx_sort_desc[0:idx_trim]

            # create dict to hold means, covariances and mixture weights
            density_estimate = {}
            density_estimate['mu_array'] = np.transpose(BGM.means_[indices_trim, :])
            density_estimate['mix_weights'] = BGM.weights_[indices_trim] / np.sum(BGM.weights_[indices_trim])

            # convert output for covariances of sklearn's BGM to full matrix format
            if density_est_opts['covariance_type'] == 'full':
                density_estimate['Sigma_array'] = BGM.covariances_[indices_trim, :, :]
            elif density_est_opts['covariance_type'] == 'diag':
                density_estimate['Sigma_array'] = np.array([np.diag(row) for row in BGM.covariances_[indices_trim, :]])
            elif density_est_opts['covariance_type'] == 'spherical':
                density_estimate['Sigma_array'] = np.tensordot(BGM.covariances_[indices_trim], np.identity(BGM.means_.shape[1]), axes = 0)

            # add to dict holding the density estimates
            density_est_dict[i] = density_estimate

        # add to dict holding the density estimates to data box
        data_box_new.set_density_estimates(density_est_dict)

        return data_box_new