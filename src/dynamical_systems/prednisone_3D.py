from ..dynamical_system import DynamicalSystem
from ..data_box import DataBox
import numpy as np
import scipy

class Prednisone3D(DynamicalSystem):

    ### attributes ###
    name = 'prednisone_3D'                                  # string with name of the system
    ROI = np.array([[0, 0.2],                               # matrix defining region of interest
                    [0, 0.2],
                    [0, 0.2]])                                      
    num_states = 2                                          # number of state variables in the model
    num_params = 3                                          # number of parameters in the model

    S0 = 2000
    k_abs = 21.3443 / S0

    ### methods: polymorphisms ###

    def __init__(self):
        super().__init__()

    def _input_func(self, t):

        t = np.atleast_1d(t)
        input_vec = np.array([self.S0 * np.exp(-self.k_abs * t), np.zeros(len(t))])

        return input_vec

    def _RHS(self, t, state, parameters):
        # 3D version of prednisone model with k_Pex = k_Lex = k_ex
        k_ex, k_PL, k_LP = parameters                                                         # unpack parameters
        A = np.array([[ -(k_ex + k_PL),           k_LP],                                      # equation for prednisone P
                      [           k_PL, -(k_ex + k_LP)]])                                     # equation for prednisolone L
            
        dx = np.dot(A, state) + self.k_abs * self._input_func(t)[:, 0]                        # calculate the RHS of the ODE
        return dx

    def solve_ODE(self, t_eval, init_cond, parameters): 
        # integrates the prednisone ODEs

        t_span = (t_eval[0], t_eval[-1])                                                                            # extract start and end time from t_eval
        sol_mat = scipy.integrate.solve_ivp(self._RHS, t_span, y0 = init_cond, t_eval=t_eval, args=[parameters])    # solve ODE using solve_ivp
        sol_mat = sol_mat.y      
                                                                                                                    # set output function (y=x in this case)
        return sol_mat
    
    def obs_func(self, sol_mat):

        y = sol_mat

        return y
    
    def prior_transform(self, u):
    # define uniform prior over ROI
        """Transforms samples `u` drawn from the unit cube to samples to those
        from our uniform prior within ROI. """

        #return np.multiply((self.ROI[:, 1] - self.ROI[:, 0]), u) + self.ROI[:, 0]
    
        return 0.2 * u

    ### methods: model-specific ###
    def gen_timeseries_data_spiral(self, gen_data_opts):
        # generate data for the prednisone spiral problem where the points on the spiral are more regular

        # number of examples
        num_examples = gen_data_opts['num_examples']

        # time array and observational noise
        t_eval = gen_data_opts['t_eval']
        obs_noise_cov = gen_data_opts['obs_noise_cov']

        # mean and std. k_ex
        mu_k_ex = gen_data_opts['mu_k_ex']
        sigma_k_ex = gen_data_opts['sigma_k_ex']
        shift_vec = gen_data_opts['shift_vec']

        # 2D covariance matrix associated with the ground truth class-distribution
        scale_spiral = gen_data_opts['scale_spiral']
        Sigma_comp = gen_data_opts['Sigma_comp']
        num_intervals = gen_data_opts['num_intervals']

        # fixed values for the Prednisone model
        init_cond = gen_data_opts['init_cond']

        # set parameters for spiral regularity
        s0 = 3
        s1 = 10
        delta = (s1 - s0) / num_intervals

        # determine center point and basis vectors for subspace
        subspace = {}
        subspace['center'] = np.concatenate(([0], shift_vec))
        subspace['v1'] = np.array([0, 1, 0])
        subspace['v2'] = np.array([0, 0, 1])

        # initialize
        labels = np.zeros(np.sum(num_examples))
        init_conds = np.zeros((np.sum(num_examples), len(init_cond)))
        parameter_mat = np.zeros((np.sum(num_examples), 3))
        timeseries_dict = {}

        cnt = 0
        for class_idx in range(len(num_examples)):
            for n in range(num_examples[class_idx]):

                # determine active interval to draw from and draw point as mean location
                interval = np.mod(n, num_intervals)
                s = np.random.uniform(s0 + interval * delta, s0 + (interval + 1) * delta, 1) 

                # sample components of k_ex randomly normal
                k_ex = np.random.normal(mu_k_ex, sigma_k_ex, 1)

                # randomly select a class label 0 or 1
                if class_idx == 0:
                    mu = scale_spiral * np.concatenate((0.04 * s * np.sin(s),  0.04 * s * np.cos(s))) + shift_vec
                else:
                    mu = scale_spiral * np.concatenate((-0.04 * s * np.sin(s),  -0.04 * s * np.cos(s))) + shift_vec

                # draw parameter from class-conditional distribution and produce timeseries
                theta_2D = np.random.multivariate_normal(mean=mu, cov=Sigma_comp)
                parameters = np.concatenate((k_ex, theta_2D))
                timeseries = self.gen_timeseries(t_eval, init_cond, parameters, obs_noise_cov)

                # update 
                labels[cnt] = class_idx
                init_conds[cnt, :] = init_cond
                parameter_mat[cnt, :] = parameters
                timeseries_dict[cnt] = timeseries

                cnt += 1

            # create DataBox and add data to it
            data_box = DataBox()

            data_box.set_labels(labels)
            data_box.set_init_conds(init_conds)
            data_box.set_parameters(parameter_mat)
            data_box.set_timeseries(timeseries_dict)

            data_box.set_dynamical_system(self)
            data_box.set_subspace(subspace)
            data_box.set_gen_data_opts(gen_data_opts)

        return data_box
    
    def gen_timeseries_data_regular_spiral(self, gen_data_opts):
        # generate data for the prednisone spiral problem where the points on the spiral are more regular

        # number of examples
        num_examples = gen_data_opts['num_examples']

        # time array and observational noise
        t_eval = gen_data_opts['t_eval']
        obs_noise_cov = gen_data_opts['obs_noise_cov']

        # range of Pex parameters
        Pex_range = gen_data_opts['Pex_range']
        shift_vec = gen_data_opts['shift_vec']

        # 2D covariance matrix associated with the ground truth class-distribution
        scale_spiral = gen_data_opts['scale_spiral']
        Sigma_comp = gen_data_opts['Sigma_comp']

        # fixed values for the Prednisone model
        init_cond = gen_data_opts['init_cond']

        # set parameters for spiral regularity
        s0 = 3
        s1 = 8           # 10

        # determine center point and basis vectors for subspace
        subspace = {}
        subspace['center'] = np.concatenate(([0], shift_vec.flatten()))
        subspace['v1'] = np.array([0, 1, 0])
        subspace['v2'] = np.array([0, 0, 1])

        # draw equally space points on spirals
        Pex_C0 = np.linspace(Pex_range[0], Pex_range[1], num_examples[0])
        Pex_C1 = np.linspace(Pex_range[0], Pex_range[1], num_examples[1])
        s_C0 = np.linspace(s0, s1, num_examples[0])
        s_C1 = np.linspace(s0, s1, num_examples[1])
        mu_C0 = scale_spiral * np.array([Pex_C0 / scale_spiral,  0.04 * s_C0 * np.sin(s_C0),  0.04 * s_C0 * np.cos(s_C0)]) + np.repeat(np.vstack((0, shift_vec)), num_examples[0], axis=1)
        mu_C1 = scale_spiral * np.array([Pex_C1 / scale_spiral, -0.04 * s_C1 * np.sin(s_C1), -0.04 * s_C1 * np.cos(s_C1)]) + np.repeat(np.vstack((0, shift_vec)), num_examples[1], axis=1)

        # initialize
        labels = np.zeros(np.sum(num_examples))
        init_conds = np.zeros((np.sum(num_examples), len(init_cond)))
        parameter_mat = np.zeros((np.sum(num_examples), 3))
        timeseries_dict = {}

        cnt = 0
        for class_idx in range(len(num_examples)):
            for n in range(num_examples[class_idx]):

                # check class label
                if class_idx == 0:
                    parameters = np.random.multivariate_normal(mean=mu_C0[:, n], cov=Sigma_comp)
                else:
                    parameters = np.random.multivariate_normal(mean=mu_C1[:, n], cov=Sigma_comp)

                # draw parameter from class-conditional distribution and produce timeseries
                timeseries = self.gen_timeseries(t_eval, init_cond, parameters, obs_noise_cov)

                # update
                labels[cnt] = class_idx
                init_conds[cnt, :] = init_cond
                parameter_mat[cnt, :] = parameters
                timeseries_dict[cnt] = timeseries

                cnt += 1

            # create DataBox and add data to it
            data_box = DataBox()

            data_box.set_labels(labels)
            data_box.set_init_conds(init_conds)
            data_box.set_parameters(parameter_mat)
            data_box.set_timeseries(timeseries_dict)

            data_box.set_dynamical_system(self)
            data_box.set_subspace(subspace)
            data_box.set_gen_data_opts(gen_data_opts)

        return data_box

    def gen_timeseries_data_rotated_spiral(self, gen_data_opts):
        # generate data for the prednisone spiral problem where the points on the spiral are more regular

        # number of examples
        num_examples = gen_data_opts['num_examples']

        # time array and observational noise
        t_lim = gen_data_opts['t_lim']
        obs_regularity = gen_data_opts['obs_regularity']
        obs_noise_cov = gen_data_opts['obs_noise_cov']

        # range of Pex parameters
        sigma_orth_comp = gen_data_opts['sigma_orth_comp']
        shift_vec = gen_data_opts['shift_vec']

        # 2D covariance matrix associated with the ground truth class-distribution
        scale_spiral = gen_data_opts['scale_spiral']

        # fixed values for the Prednisone model
        init_cond = gen_data_opts['init_cond']

        # define 3D rotation matrix
        theta_x, theta_y, theta_z = gen_data_opts['rot_angles']

        rot_mat_x = np.array([[ 1, 0           , 0             ],
                                [ 0, np.cos(theta_x),-np.sin(theta_x)],
                                [ 0, np.sin(theta_x), np.cos(theta_x)]])
        rot_mat_y =np.array([[ np.cos(theta_y), 0, np.sin(theta_y)],
                                [ 0           , 1, 0             ],
                                [-np.sin(theta_y), 0, np.cos(theta_y)]])
        rot_mat_z =np.array([[ np.cos(theta_z), -np.sin(theta_z), 0 ],
                                [ np.sin(theta_z), np.cos(theta_z) , 0 ],
                                [ 0           , 0            , 1   ]])
        rot_mat = np.dot(rot_mat_z, np.dot(rot_mat_y, rot_mat_x))

        # set parameters for spiral regularity
        s0 = 3 #3
        s1 = 8 #8           

        # determine center point and basis vectors for subspace
        subspace = {}
        subspace['center'] = shift_vec
        subspace['v1'] = np.dot(rot_mat, np.array([1., 0., 0.]))
        subspace['v2'] = np.dot(rot_mat, np.array([0., 1., 0.]))

        # draw equally space points on spirals
        s_C0 = np.linspace(s0, s1, num_examples[0])
        s_C1 = np.linspace(s0, s1, num_examples[1])
        orth_comps_C0 = np.random.normal(loc=0.0, scale=sigma_orth_comp, size=num_examples[0])
        orth_comps_C1 = np.random.normal(loc=0.0, scale=sigma_orth_comp, size=num_examples[1])
        mu_C0 = scale_spiral * np.dot(rot_mat, np.array([ 0.04 * s_C0 * np.sin(s_C0),  0.04 * s_C0 * (np.cos(s_C0) + 0.2), orth_comps_C0])) + np.repeat(shift_vec[np.newaxis], num_examples[0], axis=0).T
        mu_C1 = scale_spiral * np.dot(rot_mat, np.array([-0.04 * s_C1 * np.sin(s_C1), -0.04 * s_C1 * (np.cos(s_C1) + 0.2), orth_comps_C1])) + np.repeat(shift_vec[np.newaxis], num_examples[1], axis=0).T

        # initialize
        labels = np.zeros(np.sum(num_examples))
        init_conds = np.zeros((np.sum(num_examples), len(init_cond)))
        parameter_mat = np.zeros((np.sum(num_examples), 3))
        timeseries_dict = {}

        cnt = 0
        for class_idx in range(len(num_examples)):
            for n in range(num_examples[class_idx]):

                # check class label
                if class_idx == 0:
                    parameters = mu_C0[:, n]
                else:
                    parameters = mu_C1[:, n]

                # draw parameter from class-conditional distribution and produce timeseries
                num_t_values = np.random.randint(obs_regularity[0], obs_regularity[1] + 1)
                t_eval = np.linspace(t_lim[0], t_lim[1], num_t_values)
                timeseries = self.gen_timeseries(t_eval, init_cond, parameters, obs_noise_cov)

                # update
                labels[cnt] = class_idx
                init_conds[cnt, :] = init_cond
                parameter_mat[cnt, :] = parameters
                timeseries_dict[cnt] = timeseries

                cnt += 1

            # create DataBox and add data to it
            data_box = DataBox()

            data_box.set_labels(labels)
            data_box.set_init_conds(init_conds)
            data_box.set_parameters(parameter_mat)
            data_box.set_timeseries(timeseries_dict)

            data_box.set_dynamical_system(self)
            data_box.set_subspace(subspace)
            data_box.set_gen_data_opts(gen_data_opts)

        return data_box

    def gen_timeseries_data_rings(self, gen_data_opts):
        # generate data for the prednisone spiral problem where the points on the spiral are more regular

        # number of examples
        num_examples = gen_data_opts['num_examples']

        # time array and observational noise
        t_eval = gen_data_opts['t_eval']
        obs_noise_cov = gen_data_opts['obs_noise_cov']

        # range of Pex parameters
        mu_k_ex = gen_data_opts['mu_k_ex']
        sigma_k_ex_C0 = gen_data_opts['sigma_k_ex_C0']
        sigma_k_ex_C1 = gen_data_opts['sigma_k_ex_C1']

        shift_vec = gen_data_opts['shift_vec']

        # 2D covariance matrix associated with the ground truth class-distribution
        Sigma_comp = gen_data_opts['Sigma_comp']

        # fixed values for the Prednisone model
        init_cond = gen_data_opts['init_cond']

        radius_C0 = gen_data_opts['radius_C0']
        radius_C1 = gen_data_opts['radius_C1']

        # determine center point and basis vectors for subspace
        subspace = {}
        subspace['center'] = np.concatenate(([0], shift_vec.flatten()))
        subspace['v1'] = np.array([0, 1, 0])
        subspace['v2'] = np.array([0, 0, 1])

        # draw equally spaced points on circles
        Pex_C0 = np.random.normal(mu_k_ex, sigma_k_ex_C0, num_examples[0])
        Pex_C1 = np.random.normal(mu_k_ex, sigma_k_ex_C1, num_examples[1])
        
        # draw equally spaced points on circles
        s_C0 = np.linspace(0, 2* np.pi - 2* np.pi / num_examples[0], num_examples[0])
        s_C1 = np.linspace(0, 2* np.pi - 2* np.pi / num_examples[1], num_examples[1])
        mu_C0 = np.array([Pex_C0,  radius_C0 *  np.sin(s_C0),  radius_C0 * np.cos(s_C0)]) + np.repeat(np.vstack((0, shift_vec)), num_examples[0], axis=1)
        mu_C1 = np.array([Pex_C1,  radius_C1 *  np.sin(s_C1),  radius_C1 * np.cos(s_C1)]) + np.repeat(np.vstack((0, shift_vec)), num_examples[1], axis=1)

        # initialize
        labels = np.zeros(np.sum(num_examples))
        init_conds = np.zeros((np.sum(num_examples), len(init_cond)))
        parameter_mat = np.zeros((np.sum(num_examples), 3))
        timeseries_dict = {}

        cnt = 0
        for class_idx in range(len(num_examples)):
            for n in range(num_examples[class_idx]):

                # check class label
                if class_idx == 0:
                    parameters = np.random.multivariate_normal(mean=mu_C0[:, n], cov=Sigma_comp)
                else:
                    parameters = np.random.multivariate_normal(mean=mu_C1[:, n], cov=Sigma_comp)

                # draw parameter from class-conditional distribution and produce timeseries
                timeseries = self.gen_timeseries(t_eval, init_cond, parameters, obs_noise_cov)

                # update
                labels[cnt] = class_idx
                init_conds[cnt, :] = init_cond
                parameter_mat[cnt, :] = parameters
                timeseries_dict[cnt] = timeseries

                cnt += 1

            # create DataBox and add data to it
            data_box = DataBox()

            data_box.set_labels(labels)
            data_box.set_init_conds(init_conds)
            data_box.set_parameters(parameter_mat)
            data_box.set_timeseries(timeseries_dict)

            data_box.set_dynamical_system(self)
            data_box.set_subspace(subspace)
            data_box.set_gen_data_opts(gen_data_opts)

        return data_box
    
    def gen_timeseries_data_box(self, gen_data_opts):
        ''' generate data for the prednisone spiral problem where the points on the spiral are more regular '''

        # number of examples
        num_examples = gen_data_opts['num_examples']

        # time array and observational noise
        t_eval = gen_data_opts['t_eval']
        obs_noise_cov = gen_data_opts['obs_noise_cov']

        # mean and std. k_ex
        mu_k_ex = gen_data_opts['mu_k_ex']
        sigma_k_ex = gen_data_opts['sigma_k_ex']

        # fixed values for the Prednisone model
        init_cond = gen_data_opts['init_cond']

        # extract info about class distributions
        box_class_0 = gen_data_opts['box_class_0']
        border_width = gen_data_opts['border_width']

        # determine center point and basis vectors for subspace
        subspace = {}
        subspace['center'] = np.concatenate(([0], (box_class_0[:, 1] - box_class_0[:, 0]) / 2))
        subspace['v1'] = np.array([0, 1, 0])
        subspace['v2'] = np.array([0, 0, 1])

        # set labels and initial conditions
        labels = np.concatenate((np.zeros(num_examples[0]), np.ones(num_examples[1])))
        init_conds = np.zeros((np.sum(num_examples), len(init_cond)))
        
        # draw from inside box, draw from surround_box, stack
        kPL_kLP_inside = np.random.uniform(box_class_0[:, 0], box_class_0[:, 1], (num_examples[0], 2))
        kPL_kLP_outside = draw_from_surround_box(num_examples[1], box_class_0, border_width)
        parameter_mat_temp = np.vstack((kPL_kLP_inside, kPL_kLP_outside))

        # add random Gaussian components
        k_ex = np.random.normal(mu_k_ex, sigma_k_ex, (np.sum(num_examples), 1))

        # combine in parameter mat
        parameter_mat = np.hstack((k_ex, parameter_mat_temp))

        # generate time series
        timeseries_dict = {}
        # loop over examples

        for n in range(np.sum(num_examples)):
            init_conds[n, :] = init_cond
            timeseries = self.gen_timeseries(t_eval, init_cond, parameter_mat[n, :], obs_noise_cov)
            timeseries_dict[n] = timeseries

        # create DataBox and add data to it
        data_box = DataBox()

        data_box.set_labels(labels)
        data_box.set_init_conds(init_conds)
        data_box.set_parameters(parameter_mat)
        data_box.set_timeseries(timeseries_dict)

        data_box.set_dynamical_system(self)
        data_box.set_subspace(subspace)
        data_box.set_gen_data_opts(gen_data_opts)

        return data_box

def draw_from_surround_box(num_draw, box_class_0, border_width):

    kPL_min = box_class_0[0, 0]
    kPL_max = box_class_0[0, 1]
    kLP_min = box_class_0[1, 0]
    kLP_max = box_class_0[1, 1]

    large_num_draws = 100 * num_draw

    # draw from outside box
    cand_mat = np.random.uniform(box_class_0[:, 0] - border_width, box_class_0[:, 1] + border_width, (large_num_draws, 2))

    # discard all which are within inside box
    mask = np.ones(large_num_draws).astype(bool) 
    for i in range(large_num_draws):
        if kPL_min <= cand_mat[i, 0] and cand_mat[i, 0] <= kPL_max and kLP_min <= cand_mat[i, 1] and cand_mat[i, 1] <= kLP_max:
            mask[i] = False

    # collect the ones in the outside box
    in_outside_box = cand_mat[mask, :]

    # pick equally many from 
    
    return in_outside_box[:num_draw, :]