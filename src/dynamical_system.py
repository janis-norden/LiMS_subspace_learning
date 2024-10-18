import numpy as np
import scipy
import src

class DynamicalSystem:

    name = None                                     # string with name of the system
    ROI = None                                      # matrix defining region of interest
    num_states = None                               # number of state variables in the model
    num_params = None                               # number of parameters in the model

    def __init__(self):
        # to be specified in class specific to the dynamical model at hand
        pass            

    def _input_func(self, t):
        pass

    def _RHS(self, t, state, parameters):
        # to be specified in class specific to the dynamical model at hand
        pass

    def solve_ODE(self, t_eval, init_cond, parameters):
        # to be specified in class specific to the dynamical model at hand
        pass                  

    def obs_func(self, sol_mat):
        # to be specified in class specific to the dynamical model at hand
        pass

    def gen_timeseries(self, t_eval, init_cond, parameters, obs_noise_cov):
        # Solve ODE and add noise to the components as specified in cov_mat

        # solve ODE
        sol_mat = self.solve_ODE(t_eval, init_cond, parameters)

        # apply observation function
        sol_mat_obs = self.obs_func(sol_mat)

        # construct noise matrix and add to observable
        mean = np.zeros(np.size(obs_noise_cov, 0))
        num_time_points = np.size(sol_mat_obs, 1)
        obs_noise_mat = np.random.multivariate_normal(mean, obs_noise_cov, size = num_time_points)
        sol_mat_obs_noise = sol_mat_obs + np.transpose(obs_noise_mat)

        # collect t_eeval and observable with added noise in single matrix
        timeseries = np.vstack((t_eval.reshape(1, len(t_eval)), sol_mat_obs_noise))

        return timeseries

    def loglikelihood(self, init_cond, parameters, timeseries, obs_noise_cov):

        # evaluate the loglikelihood of the observed data, given the parameter vector, covariance matrix of the observational noise is assumed to be known
        
        # extract times at which timeseries is evaulated
        t_eval = timeseries[0, :]        
        num_time_points = len(t_eval)

        # extract state observations from timeseries               
        timeseries_obs = timeseries[1:, :]      

        # solve the ODE
        sol_mat = self.solve_ODE(t_eval, init_cond, parameters)    

        # apply observation function
        sol_mat_obs = self.obs_func(sol_mat)

        # evaluate Gaussian deviation at each observation point 
        num_dim = np.shape(timeseries_obs)[0]
        loglike_vec = np.zeros(num_time_points)
        norm_const = -0.5 * (np.log(2 * np.pi) * num_dim + np.log(np.linalg.det(obs_noise_cov)))  # normalization factor for Gaussian
        cov_mat_inv = np.linalg.inv(obs_noise_cov)
        
        # loop over time points
        for i in range(num_time_points):
            loglike_vec[i] = norm_const - 0.5 * np.dot(timeseries_obs[:, i] - sol_mat_obs[:, i], np.dot(cov_mat_inv, timeseries_obs[:, i] - sol_mat_obs[:, i]))
        loglike_value = sum(loglike_vec)

        return loglike_value