import numpy as np
import dill
import dynesty
import dynesty.utils
dynesty.utils.pickle_module = dill
from multiprocessing import Pool
from functools import partial
from copy import copy

class Sampler:

    ### attributes ###
    data_box = None
    sampler_opts = None

    ### methods ###
    def __init__(self, data_box, sampler_opts):
        self.data_box = copy(data_box)
        self.sampler_opts = copy(sampler_opts)  

    def _loglikelihood_select(self, i, parameters):
        # define loglikelihood function for example i
        value = self.data_box._dynamical_system.loglikelihood(init_cond = self.data_box._init_conds[i], 
                                                            parameters = parameters, 
                                                            timeseries = self.data_box._timeseries[i] , 
                                                            obs_noise_cov = self.data_box._gen_data_opts['obs_noise_cov'])
        return value

    def run_sampling(self):
        # execute sampling with configurations specified in sampler_opts

        # check sampling type wanted
        if self.sampler_opts['type'] == 'nested_sampling':
            data_box = self._run_nested_sampling()
        
        # store sampler_opts in data_box
        data_box._sampler_opts = self.sampler_opts

        return data_box
    
    def _run_nested_sampling(self):
        # runs Nested Sampling by calling dynesty

        # extract from self
        data_box = self.data_box
        sampler_opts = self.sampler_opts
        dynamical_system = data_box._dynamical_system

        # extract prior transform from dynamical_system
        prior_transform = dynamical_system.prior_transform

        # initialize dict
        posterior_samples = {}

        # check if parallel computation is wanted
        if sampler_opts['parallel'] == True:

            # loop over examples
            for i in range(0, len(data_box._labels)):
                
                # perform nested sampling in parallel
                with Pool(sampler_opts['num_CPU']) as pool:
                    sampler = dynesty.NestedSampler(partial(self._loglikelihood_select, i), 
                                                    prior_transform, 
                                                    pool = pool,
                                                    ndim = dynamical_system.num_params,
                                                    nlive = sampler_opts['num_live_points'], 
                                                    queue_size = sampler_opts['num_CPU']
                                                    )
                    sampler.run_nested()
                    
                # extract posterior samples from dynesty sampler
                posterior_samples[i] = self._extract_samples(sampler_opts, sampler)
        else:

            # loop over examples
            for i in range(0, len(data_box._labels)):
                
                # perform nested sampling
                sampler = dynesty.NestedSampler(partial(self._loglikelihood_select, i), 
                                                    prior_transform,
                                                    ndim = dynamical_system.num_params, 
                                                    nlive = sampler_opts['num_live_points'], 
                                                    )
                sampler.run_nested()

                # extract posterior samples from dynesty sampler
                posterior_samples[i] = self._extract_samples(sampler_opts, sampler)
        
        # add posterior samples to data box
        data_box.set_posterior_samples(posterior_samples)

        return data_box
    
    def _extract_samples(self, sampler_opts, sampler):
            # extracts posterior samples from dynesty sampler object
            # provide weighted or unweighted samples

            posterior_samples = {}
            if sampler_opts['weighted'] == True:
                posterior_samples['samples'] = sampler.results.samples
                posterior_samples['weights'] = sampler.results.importance_weights()
            else:
                posterior_samples['samples'] = sampler.results.samples_equal()
                posterior_samples['weights'] = (1 / np.shape(sampler.results.samples_equal())[0]) * np.ones(np.shape(sampler.results.samples_equal())[0])
            
            return posterior_samples