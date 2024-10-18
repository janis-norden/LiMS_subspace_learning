import numpy as np
import logging
import scipy
from sklvq import GMLVQ
from numba import njit, prange
from datetime import datetime
from src.utility import create_masked_arrays, set_class_prior
import src.config as config
from copy import copy

class SubspaceLearner:

    ### attributes ###
    data_box = None
    subspace_learn_opts = None

    ### methods ###
    def __init__(self, data_box, subspace_learn_opts):
        self.data_box = copy(data_box)
        self.subspace_learn_opts = copy(subspace_learn_opts)

    def run_subspace_learning(self):
        # execute subspace learning with configurations specified in subspace_learn_opts
        
        # log call to subspace learner
        logging.info('--- Call to subspace learner ---')

        # unpack from self
        data_box = self.data_box
        subspace_learn_opts = self.subspace_learn_opts

        # set initial subspace
        V0 = self._initialize_subspace()

        # set class prior
        class_prior = set_class_prior(data_box.get_labels(), subspace_learn_opts['class_prior'])

        # create masked data arrays
        masked_arrays = create_masked_arrays(data_box)

        # select cost function
        if subspace_learn_opts['cost_function'] == 'subspace_likelihood':
            
            # define cost function
            cost_fun = lambda V_flat, shape_V : self._subspace_neg_loglikelihood(V_flat, shape_V, class_prior, masked_arrays)
            
            # no jacobian availabe -> set to False
            subspace_learn_opts['jacobian'] = None

        elif subspace_learn_opts['cost_function'] == 'subspace_likelihood_gradient':

            # define cost function
            cost_fun = lambda V_flat, shape_V : self._subspace_neg_loglikelihood_and_gradient(V_flat, shape_V, class_prior, masked_arrays)
            
            # jacobian availabe -> set to True
            subspace_learn_opts['jacobian'] = True


        elif subspace_learn_opts['cost_function'] == 'GLVQ':

            # find Gaussian mixtures of class-conditionals
            masked_arrays_cc = self._find_class_conditionals_masked()

            # extract density estimates
            #density_estimates = data_box.get_density_estimates()

            # define cost function
            cost_fun = lambda V_flat, shape_V : self._GLVQ_cost_masked(V_flat, shape_V, masked_arrays, masked_arrays_cc)

            # no jacobian availabe -> set to False
            subspace_learn_opts['jacobian'] = None

        # run selected optimization type
        if subspace_learn_opts['opt_mode'] == 'batch':
            subspace_learned = self._run_batch_optimization(cost_fun, V0)
        elif subspace_learn_opts['opt_mode'] == 'iterative':
            subspace_learned =self._run_iterative_optimization(cost_fun, V0)
            pass

        # store subspace_learned and subspace_learn_opts in data_box
        data_box.set_subspace_learned(subspace_learned)
        data_box.set_subspace_learn_opts(subspace_learn_opts)
            
        return data_box
    
    def _set_initial_condition_iterative(self, V0, Q):
        ''' finds a projection of the wanted initial condition V0 onto the row space of Q'''

        # extract the number of columns of V0
        num_cols_V0 = V0.shape[1]

        # loop over the columns of V0
        for j in range(num_cols_V0):

            # select jth column of V0
            cand_column = V0[:, j]

            # check if cand_column is orthognal to Q
            cand_V0_iter = np.dot(Q.T, cand_column)
            if np.linalg.norm(cand_V0_iter) > 10^-6:
                break

        # assign output
        V0_iter = cand_V0_iter

        return V0_iter

    def _initialize_subspace(self):
        ''' returns matrix V0 used as initial matrix for optimization '''
        
        # unpack from self
        subspace_learn_opts = self.subspace_learn_opts

        # check which initialization type is wanted
        if isinstance(subspace_learn_opts['init_type'], str) and subspace_learn_opts['init_type'] == 'PCA':

            # merge all samples together
            samples_all = self.data_box.merge_samples_all()

            # perform SVD on samples_all
            U, S, Vh = np.linalg.svd(np.transpose(samples_all), full_matrices=False)

            # extract dominant subspace_dim directions of U
            V0 = np.copy(U[:, 0:subspace_learn_opts['subspace_dim']])

        elif isinstance(subspace_learn_opts['init_type'], str) and subspace_learn_opts['init_type'] == 'random':
            
            # extract dimension of ambient space
            dynamical_system = self.data_box.get_dynamical_system()
            ambient_dim = dynamical_system.num_params

            # generate random initial matrix V
            V0 = np.random.rand(ambient_dim, subspace_learn_opts['subspace_dim'])

        elif isinstance(subspace_learn_opts['init_type'], str) and subspace_learn_opts['init_type'] == 'GMLVQ':
            
            # merge all means from all mixtures together
            means_merged, labels_merged = self.data_box.merge_means_and_labels()

            # create model object to fit data to
            model = GMLVQ(
                distance_type="adaptive-squared-euclidean",
                activation_type="swish",
                activation_params={"beta": 2},
                solver_type="waypoint-gradient-descent",
                solver_params={"max_runs": 50, "k": 10, "step_size": np.array([0.1, 0.05])}
            )

            # train the model and extract relevance matrix
            model.fit(means_merged, labels_merged)
            relevance_matrix = model.lambda_

            # perform SVD on relevance_matrix
            U, S, Vh = np.linalg.svd(relevance_matrix, full_matrices=False)

            # extract dominant subspace_dim directions of relevance_matrix
            V0 = np.copy(U[:, 0:subspace_learn_opts['subspace_dim']])

        else:

            # set V0 to the matrix provided
            V0 = subspace_learn_opts['init_type']

        return V0

    # subspace likelihood methods
    def _subspace_neg_loglikelihood(self, V_flat, shape_V, class_prior, masked_arrays):
        ''' evaluates the subspace likelihood '''
        
        # unfold vector VFlat into matrix V
        V = np.copy(V_flat.reshape(shape_V))

        # extract masked arrays
        labels = masked_arrays['labels']
        num_samples_vec = masked_arrays['num_samples_vec']
        num_comps_vec = masked_arrays['num_comps_vec']

        samples_mask = masked_arrays['samples']
        sample_weights_mask = masked_arrays['sample_weights']

        mu_array_mask = masked_arrays['mu_array']
        Sigma_array_mask = masked_arrays['Sigma_array']
        mix_weights_mask = masked_arrays['mix_weights']

        num_pat_per_class_vec = masked_arrays['num_pat_per_class_vec']
        class_idx_mat = masked_arrays['class_idx_mat']

        # check covariance structure
        if self.data_box.get_density_est_opts()['covariance_type'] == 'full':

            if self.subspace_learn_opts['ssl_approx'] == False:
                # call to NUMBA implemention
                value, f_array, g_array, n_sum_vec = self._bottleneck_full(V, class_prior, num_samples_vec, num_comps_vec, samples_mask, sample_weights_mask, mu_array_mask, Sigma_array_mask, mix_weights_mask, labels, num_pat_per_class_vec, class_idx_mat)
            else:
                # call to NUMBA implemention
                value, f_array, g_array, n_sum_vec = self._bottleneck_full_approx(V, class_prior, num_samples_vec, num_comps_vec, samples_mask, sample_weights_mask, mu_array_mask, Sigma_array_mask, mix_weights_mask, labels, num_pat_per_class_vec, class_idx_mat)
        
        elif self.data_box.get_density_est_opts()['covariance_type'] == 'diag':
            # call to NUMBA implemention NOTE:  this simply call s the same routine as for 'full'
            value, f_array, g_array, n_sum_vec = self._bottleneck_full(V, class_prior, num_samples_vec, num_comps_vec, samples_mask, sample_weights_mask, mu_array_mask, Sigma_array_mask, mix_weights_mask, labels, num_pat_per_class_vec, class_idx_mat)
        
        elif self.data_box.get_density_est_opts()['covariance_type'] == 'spherical':
            # call to NUMBA implemention
            value, f_array, g_array, n_sum_vec = self._bottleneck_spherical(V, class_prior, num_samples_vec, num_comps_vec, samples_mask, sample_weights_mask, mu_array_mask, Sigma_array_mask, mix_weights_mask, labels, num_pat_per_class_vec, class_idx_mat)

        return -value

    #   batch
    def _run_batch_optimization(self, cost_fun, V0):
        '''run optimization for the entire projection matrix V in one go'''

        # extract from self
        subspace_learn_opts = self.subspace_learn_opts

        # extract shape of V
        shape_V = np.shape(V0)

        # initialize global variable to pass to callback function   (NEW)
        global OPT_INFO
        OPT_INFO = config.OPT_INFO
        OPT_INFO['opt_iter'] = 0
        OPT_INFO['shape_V'] = shape_V
        OPT_INFO['opt_history'] = {'fun': [], 'V': [], 'time': []}

        # set cost function for batch optimization
        cost_fun_batch_opt = lambda V_flat : cost_fun(V_flat, shape_V)

        # optimize cost function
        opt_results = scipy.optimize.minimize(cost_fun_batch_opt, 
                                V0.flatten(), 
                                method = subspace_learn_opts['scipy_min_method'], 
                                tol = subspace_learn_opts['scipy_min_tol'], 
                                options = {'maxiter': subspace_learn_opts['scipy_min_maxiter'], 
                                          'disp': subspace_learn_opts['scipy_min_disp'] },
                                jac = subspace_learn_opts['jacobian'],
                                callback = self._callback_func_batch)
        
        # evaluate cost at V0
        fun0 = cost_fun_batch_opt(V0.flatten())

        # create dict to hold information about learned subspace
        subspace_learned = {
            'V0': V0,
            'fun0': fun0,
            'V_opt':        np.reshape(opt_results.x, shape_V),
            'cost':         opt_results.fun,
            'message':      opt_results.message,
            'opt_history':  OPT_INFO['opt_history']
            }
        
        return subspace_learned

    def _callback_func_batch(self, intermediate_result):
        ''' callback function for optimizer gets called after each iteration of the optimization process'''
        
        global OPT_INFO
    
        # get model name, dim of subspace and optimization iteration
        model_name = self.data_box.get_dynamical_system().name
        shape_V = OPT_INFO['shape_V']
        opt_iter = OPT_INFO['opt_iter']
        fold =  OPT_INFO['fold']

        # find dimension of desired subspace
        subspace_dim = shape_V[1]
        
        # get current V and cost function value
        current_fun_val = intermediate_result.fun
        current_V = np.reshape(intermediate_result.x, shape_V)

        # get current time
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        # print subspace learning update to command line
        print(
            'Model:', model_name, ' | ',
            'Subspace dim.:', subspace_dim, ' | ',
            'Fold:', fold, ' | ',
            'Opt. type: batch', ' | ',
            'Opt. iter:', str(opt_iter).zfill(2),  ' | ', 
            'Cost:', f'{current_fun_val:.8f}', ' | '
            'Time:', current_time
            )
        
        # append values to optimization history dict and update optimization iteration counter
        OPT_INFO['opt_history']['fun'].append(current_fun_val)
        OPT_INFO['opt_history']['V'].append(current_V)
        OPT_INFO['opt_history']['time'].append(current_time)
        OPT_INFO['opt_iter'] += 1

    #   iterative
    def _run_iterative_optimization(self, cost_fun, V0):
        '''run optimization for columns of projection matrix V iteratively'''

        # extract from self
        subspace_learn_opts = self.subspace_learn_opts

        # extract the number of required columns
        num_rows_V, num_cols_V = V0.shape

        # initialize global variable to pass to callback function       # NEW
        global OPT_INFO
        OPT_INFO = config.OPT_INFO
        OPT_INFO['col_num'] =  1
        OPT_INFO['opt_iter'] = 0
        OPT_INFO['Q'] = None
        OPT_INFO['last_opt_history'] = None
        OPT_INFO['opt_histories'] = []

        # construct columns of V iteratively
        for i in range(num_cols_V):

            # assess required shaped of V at the current iteration
            shape_V = np.array([num_rows_V, i + 1])

            OPT_INFO['opt_iter'] = 0
            OPT_INFO['last_opt_history'] = {'fun': [], 'V': [], 'time': []}

            # find orthonorm. basis Q for ortho. complement of span(V_opt) aka. null(V_opt^T)
            if i == 0:
                Q = np.identity(num_rows_V)
                cost_func_iter = lambda V_flat : cost_fun(V_flat, shape_V)
            else:
                Q = scipy.linalg.null_space(np.transpose(V_opt), rcond=1e-18)
                cost_func_iter = lambda Q_coords : cost_fun(np.hstack((V_opt, np.dot(Q, np.reshape(Q_coords, (num_rows_V - i, 1))))), shape_V) 

            # find suitable inital value for optimization in the row space of Q
            V0_iter = self._set_initial_condition_iterative(V0, Q)

            # add Q to OPT_INFO
            OPT_INFO['Q'] = Q

            # optimize cost function in basis of Q
            opt_results = scipy.optimize.minimize(cost_func_iter, 
                                V0_iter.flatten(), 
                                method = subspace_learn_opts['scipy_min_method'], 
                                tol = subspace_learn_opts['scipy_min_tol'], 
                                options = {'maxiter': subspace_learn_opts['scipy_min_maxiter'], 
                                            'disp': subspace_learn_opts['scipy_min_disp'] },
                                jac = subspace_learn_opts['jacobian'],
                                callback = self._callback_func_iterative)

            # normalize the found optimal column
            if i == 0:
                V_opt = np.reshape(opt_results.x, (num_rows_V, 1))
                V_opt = V_opt / np.linalg.norm(V_opt)
            else:
                V_new = np.dot(Q, np.reshape(opt_results.x, (num_rows_V - i, 1)))
                V_new = V_new / np.linalg.norm(V_new)
                V_opt = np.hstack((V_opt, V_new))

            # update OPT_INFO
            OPT_INFO['opt_histories'].append(OPT_INFO['last_opt_history'])
            OPT_INFO['col_num'] += 1
        
        # create dict to hold information about learned subspace
        subspace_learned = {
            'V0': V0,
            'V_opt':        V_opt,
            'cost':         opt_results.fun,
            'message':      opt_results.message,
            'opt_histories':  OPT_INFO['opt_histories']
            }
        
        return subspace_learned

    def _callback_func_iterative(self, intermediate_result):
        ''' callback function for optimizer gets called after each iteration of the optimization process'''
        
        global OPT_INFO
    
        # get model name, and dim. of subspace 
        model_name = self.data_box.get_dynamical_system().name
        subspace_dim = self.subspace_learn_opts['subspace_dim']
        
        # get optimization iteration, # of current column, Q matrix
        opt_iter = OPT_INFO['opt_iter']
        col_num = OPT_INFO['col_num']
        Q = OPT_INFO['Q'] 
        fold =  OPT_INFO['fold']
        
        # get current V and cost function value
        current_fun_val = intermediate_result.fun
        current_V = np.dot(Q, intermediate_result.x)

        # get current time
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        # print subspace learning update to command line
        print('Model:', model_name, ' | ', 
            'Subspace dim.:', subspace_dim, ' | ',
            'Fold:', fold, ' | ',
            'Opt. type: iterative', ' | ',
            'Column #:', col_num, ' | ',
            'Opt. iter:', str(opt_iter).zfill(2),  ' | ', 
            'Cost:', f'{current_fun_val:.8f}', ' | '
            'Time:', current_time)
        
        # append values to optimization history dict and update optimization iteration counter
        OPT_INFO['last_opt_history']['fun'].append(current_fun_val)
        OPT_INFO['last_opt_history']['V'].append(current_V)
        OPT_INFO['last_opt_history']['time'].append(current_time)
        OPT_INFO['opt_iter'] += 1

    @staticmethod
    @njit(parallel=True)
    def _bottleneck_spherical(V, class_prior, num_samples_vec, num_comps_vec, samples, sample_weights, mu_array, Sigma_array, mix_weights, labels, num_pat_per_class_vec, class_idx_mat):
        '''NUMBA implementation of computational bottleneck for spherical covariance matrices'''   

        # find number of patients and classes
        num_examples = np.shape(labels)[0]
        num_classes = np.shape(class_prior)[0]
        num_col_V = np.shape(V)[1]

        # find determinant and inverse of V^T * V
        det_VtV = np.linalg.det(np.dot(np.transpose(V), V))
        inv_VtV = np.linalg.inv(np.dot(np.transpose(V), V))
        
        f_array = np.zeros((num_examples, samples.shape[2]))
        g_array = np.zeros((num_examples, samples.shape[2]))
        n_sum_vec = np.zeros(num_examples)
        #value = 0.

        for n in prange(num_examples):
            #s_sum_val = 0.
            num_samples = num_samples_vec[n]

            for s in range(num_samples):
                
                c_sum_vec = np.zeros(num_classes)

                for c in range(num_classes):

                    m_k_sum_val = 0.

                    idx_label_c = class_idx_mat[0:num_pat_per_class_vec[c], c]
                    num_pat_in_class_c = num_pat_per_class_vec[c]

                    for mm in range(num_pat_in_class_c):
                        m = idx_label_c[mm]
                        num_mix_comp = num_comps_vec[m]
                        
                        for k in range(num_mix_comp):

                            theta = samples[n, :, s]
                            mu = mu_array[m, :, k]
                            sigma_sq = Sigma_array[m, k]
                            mix_weight = mix_weights[m, k]

                            a = theta - mu
                            Vta = np.dot(np.transpose(V), a)
                            det_VtSigmaV = (sigma_sq ** num_col_V) * det_VtV
                            inv_VtSigmaVVta = np.dot((1 / sigma_sq) * inv_VtV, Vta)
                            mvn_pdf_eval = (2 * np.pi) ** (-num_col_V / 2) * det_VtSigmaV ** (-0.5) * np.exp(-0.5 * np.dot(np.transpose(Vta), inv_VtSigmaVVta))
            
                            m_k_sum_val += mix_weight * mvn_pdf_eval

                    m_k_sum_val /= num_pat_in_class_c
                    c_sum_vec[c] = m_k_sum_val * class_prior[c]

                alpha = sample_weights[n, s]
                #s_sum_val += alpha * (c_sum_vec[labels[n]] / np.sum(c_sum_vec))

                f_array[n, s] = alpha * c_sum_vec[labels[n]]
                g_array[n, s] = np.sum(c_sum_vec)

            n_sum_vec[n] = np.sum(f_array[n, :num_samples] / g_array[n, :num_samples])
            #value += np.log(s_sum_val)
        
        value = np.sum(np.log(n_sum_vec))

        return value, f_array, g_array, n_sum_vec

    @staticmethod
    @njit(parallel=True)
    def _bottleneck_full(V, class_prior, num_samples_vec, num_comps_vec, samples, sample_weights, mu_array, Sigma_array, mix_weights, labels, num_pat_per_class_vec, class_idx_mat):
        '''NUMBA implementation of computational bottleneck for full covariance matrices'''                              
        
        # find number of examples and classes
        num_examples = np.shape(labels)[0]
        num_classes = np.shape(class_prior)[0]
        num_col_V = np.shape(V)[1]
        
        f_array = np.zeros((num_examples, samples.shape[2]))
        g_array = np.zeros((num_examples, samples.shape[2]))
        n_sum_vec = np.zeros(num_examples)
        #value = 0.

        for n in prange(num_examples):
            #s_sum_val = 0.
            num_samples = num_samples_vec[n]

            for s in range(num_samples):
                
                c_sum_vec = np.zeros(num_classes)

                for c in range(num_classes):

                    m_k_sum_val = 0.
                    idx_label_c = class_idx_mat[0:num_pat_per_class_vec[c], c]
                    num_pat_in_class_c = num_pat_per_class_vec[c]

                    for mm in range(num_pat_in_class_c):
                        m = idx_label_c[mm]
                        num_mix_comp = num_comps_vec[m]
                        
                        for k in range(num_mix_comp):

                            theta = samples[n, :, s]
                            mu = mu_array[m, :, k]
                            Sigma = np.copy(Sigma_array[m, k, :, :])                    # NOTE this could potentially be improved
                            mix_weight = mix_weights[m, k]

                            a = theta - mu
                            Vta = np.dot(np.transpose(V), a)
                            VtSigmaV = np.dot(np.transpose(V), np.dot(Sigma, V))

                            det_VtSigmaV = np.linalg.det(VtSigmaV)
                            inv_VtSigmaVVta = np.linalg.solve(VtSigmaV, Vta)

                            mvn_pdf_eval = (2 * np.pi) ** (-num_col_V / 2) * det_VtSigmaV ** (-0.5) * np.exp(-0.5 * np.dot(np.transpose(Vta), inv_VtSigmaVVta))
            
                            m_k_sum_val += mix_weight * mvn_pdf_eval

                    m_k_sum_val /= num_pat_in_class_c
                    c_sum_vec[c] = m_k_sum_val * class_prior[c]

                alpha = sample_weights[n, s]
                #s_sum_val += alpha * (c_sum_vec[labels[n]] / np.sum(c_sum_vec))

                f_array[n, s] = alpha * c_sum_vec[labels[n]]
                g_array[n, s] = np.sum(c_sum_vec)

            n_sum_vec[n] = np.sum(f_array[n, :num_samples] / g_array[n, :num_samples])
            #value += np.log(s_sum_val)
        
        value = np.sum(np.log(n_sum_vec))

        return value, f_array, g_array, n_sum_vec
    
    @staticmethod
    @njit(parallel=True)
    def _bottleneck_full_approx(V, class_prior, num_samples_vec, num_comps_vec, samples, sample_weights, mu_array, Sigma_array, mix_weights, labels, num_pat_per_class_vec, class_idx_mat):
        '''NUMBA implementation of computational bottleneck for full covariance matrices'''                              
        
        # find number of examples and classes
        num_examples = np.shape(labels)[0]
        num_classes = np.shape(class_prior)[0]
        num_col_V = np.shape(V)[1]
        
        f_array = np.zeros((num_examples, samples.shape[2]))
        g_array = np.zeros((num_examples, samples.shape[2]))
        n_sum_vec = np.zeros(num_examples)
        #value = 0.

        for n in prange(num_examples):
            #s_sum_val = 0.
            num_samples = num_samples_vec[n]

            for s in range(num_samples):
                
                c_sum_vec = np.zeros(num_classes)

                for c in range(num_classes):

                    m_k_sum_val = 0.
                    idx_label_c = class_idx_mat[0:num_pat_per_class_vec[c], c]
                    num_pat_in_class_c = num_pat_per_class_vec[c]

                    for mm in range(num_pat_in_class_c):
                        m = idx_label_c[mm]
                        num_mix_comp = num_comps_vec[m]
                        
                        for k in range(num_mix_comp):

                            theta = samples[n, :, s]
                            mu = mu_array[m, :, k]
                            Sigma = np.copy(Sigma_array[m, k, :, :])                    # NOTE this could potentially be improved
                            mix_weight = mix_weights[m, k]

                            a = theta - mu
                            Vta = np.dot(np.transpose(V), a)
                            VtSigmaV_approx_diag = np.diag(np.dot(np.transpose(V), np.dot(Sigma, V)))
                            #VtSigmaV_approx = np.diag(VtSigmaV_approx_diag)

                            det_VtSigmaV = np.prod(VtSigmaV_approx_diag)
                            inv_VtSigmaVVta = np.dot(np.diag(1 / VtSigmaV_approx_diag), Vta)

                            mvn_pdf_eval = (2 * np.pi) ** (-num_col_V / 2) * det_VtSigmaV ** (-0.5) * np.exp(-0.5 * np.dot(np.transpose(Vta), inv_VtSigmaVVta))
            
                            m_k_sum_val += mix_weight * mvn_pdf_eval

                    m_k_sum_val /= num_pat_in_class_c
                    c_sum_vec[c] = m_k_sum_val * class_prior[c]

                alpha = sample_weights[n, s]
                #s_sum_val += alpha * (c_sum_vec[labels[n]] / np.sum(c_sum_vec))

                f_array[n, s] = alpha * c_sum_vec[labels[n]]
                g_array[n, s] = np.sum(c_sum_vec)

            n_sum_vec[n] = np.sum(f_array[n, :num_samples] / g_array[n, :num_samples])
            #value += np.log(s_sum_val)
        
        value = np.sum(np.log(n_sum_vec))

        return value, f_array, g_array, n_sum_vec

    # GLVQ inspired cost
    def _GLVQ_cost(self, V_flat, shape_V, density_estimates, class_conditionals):
        '''heuristic cost function derived from the distance-based GLVQ cost function'''

        # unfold vector VFlat into matrix V
        V = np.copy(V_flat.reshape(shape_V))

        # define nonlinear Phi-function
        Phi_func = lambda x: x

        # find distance of every example mixture to the every class-condtional mixture
        dist_2_correct, dist_2_close_incorrect = self._calc_dists_to_class_conds(V, density_estimates, class_conditionals)

        # calculate GLVQ-type distance-based cost function
        errors = (dist_2_correct - dist_2_close_incorrect) / (dist_2_correct + dist_2_close_incorrect)
        cost = np.sum(Phi_func(errors))

        return cost

    # GLVQ inspired cost
    def _GLVQ_cost_masked(self, V_flat, shape_V, masked_arrays, masked_arrays_cc):
        '''heuristic cost function derived from the distance-based GLVQ cost function'''

        # extract masked arrays for examples
        labels = masked_arrays['labels']
        num_comps_vec = masked_arrays['num_comps_vec']

        mu_array_mask = masked_arrays['mu_array']
        Sigma_array_mask = masked_arrays['Sigma_array']
        mix_weights_mask = masked_arrays['mix_weights']

        num_pat_per_class_vec = masked_arrays['num_pat_per_class_vec']

        # extract masked arrays for class conditionals
        num_comps_vec_cc = masked_arrays_cc['num_comps_vec']

        mu_array_mask_cc = masked_arrays_cc['mu_array']
        Sigma_array_mask_cc = masked_arrays_cc['Sigma_array']
        mix_weights_mask_cc = masked_arrays_cc['mix_weights']

        # unfold vector VFlat into matrix V
        V = np.copy(V_flat.reshape(shape_V))

        # if covariances are spherical, adjust Sigma_array_mask to be of right dimensions
        if self.data_box.get_density_est_opts()['covariance_type'] == 'spherical':
            #Sigma_array_mask_adjusted = np.tensordot(Sigma_array_mask, np.identity(mu_array_mask.shape[1]), axes = 0)
            dist_2_correct, dist_2_close_incorrect = _calc_dists_to_class_conds_masked_spherical(V, 
                                                                                        labels, 
                                                                                        num_comps_vec, 
                                                                                        mu_array_mask, 
                                                                                        Sigma_array_mask, 
                                                                                        mix_weights_mask, 
                                                                                        num_pat_per_class_vec, 
                                                                                        num_comps_vec_cc, 
                                                                                        mu_array_mask_cc, 
                                                                                        Sigma_array_mask_cc, 
                                                                                        mix_weights_mask_cc)
        else:
            dist_2_correct, dist_2_close_incorrect = _calc_dists_to_class_conds_masked_full(V, 
                                                                                        labels, 
                                                                                        num_comps_vec, 
                                                                                        mu_array_mask, 
                                                                                        Sigma_array_mask, 
                                                                                        mix_weights_mask, 
                                                                                        num_pat_per_class_vec, 
                                                                                        num_comps_vec_cc, 
                                                                                        mu_array_mask_cc, 
                                                                                        Sigma_array_mask_cc, 
                                                                                        mix_weights_mask_cc)

        

        # define nonlinear Phi-function
        Phi_func = lambda x: x
        
        # calculate GLVQ-type distance-based cost function
        errors = (dist_2_correct - dist_2_close_incorrect) / (dist_2_correct + dist_2_close_incorrect)
        cost = np.sum(Phi_func(errors))

        return cost

    def _find_class_conditionals(self):

        # extract from data box
        data_box = self.data_box
        density_estimates = data_box.get_density_estimates()

        labels = data_box.get_labels()
        num_classes = len(np.unique(labels))
        num_dim = density_estimates[0]['mu_array'].shape[0]

        class_conditionals = {}
        for c in range(num_classes):

            # find indices of patients in class c
            class_indices = np.array([idx for idx, value in enumerate(labels) if value == c])

            # initialize arrays
            class_cond_mu_array = np.zeros((num_dim, 1))                                      # TODO find more elegant solution for this concatenation
            class_cond_Sigma_array = np.zeros((1, num_dim, num_dim))
            class_cond_mix_weights = np.zeros(1)

            for n in class_indices:
                class_cond_mu_array = np.append(class_cond_mu_array, density_estimates[n]['mu_array'], axis = 1)
                class_cond_Sigma_array = np.append(class_cond_Sigma_array, density_estimates[n]['Sigma_array'], axis = 0)
                class_cond_mix_weights = np.append(class_cond_mix_weights, density_estimates[n]['mix_weights'], axis = 0)
            
            # pop off slice used for initialization
            class_cond_mu_array = class_cond_mu_array[:, 1:]
            class_cond_Sigma_array = class_cond_Sigma_array[1:, :, :]
            class_cond_mix_weights = class_cond_mix_weights[1:]

            class_cond_dist = {}
            class_cond_dist['mu_array'] = class_cond_mu_array
            class_cond_dist['Sigma_array'] = class_cond_Sigma_array
            class_cond_dist['mix_weights'] = class_cond_mix_weights

            class_conditionals[c] = class_cond_dist

        return class_conditionals

    def _find_class_conditionals_masked(self):
        ''' extract data from data box and collect class conditionals into masked np.arrays '''

        # extract from data box
        labels = self.data_box.get_labels().astype(int)
        density_estimates = self.data_box.get_density_estimates()
        num_params = self.data_box.get_dynamical_system().num_params

        # determine how many patients there are per class and the maximum number of patients in a single class
        unique, counts = np.unique(labels, return_counts=True)
        num_pat_per_class_vec = counts
        num_classes = len(unique)

        # put patient indices into single matrix
        max_pat_in_class = np.max(num_pat_per_class_vec)
        class_idx_mat = np.zeros((max_pat_in_class, num_classes), dtype=int)
        for c in range(num_classes):
            class_idx_mat[0:num_pat_per_class_vec[c], c] = [idx for idx, value in enumerate(labels) if value == c]

        # find maximal number of components present in a single class
        num_comps_cc_vec = np.zeros(num_classes, dtype=int)
        for c in range(num_classes):
            num_comps_class_c = 0
            for n in range(num_pat_per_class_vec[c]):
                num_comps_class_c += len(density_estimates[class_idx_mat[n, c]]['mix_weights'])
            num_comps_cc_vec[c] = num_comps_class_c
        max_num_comps_cc = np.max(num_comps_cc_vec)

        # initialize masked arrays
        mu_array_masked = np.zeros((num_classes, num_params, max_num_comps_cc))
        mix_weights_masked = np.zeros((num_classes, max_num_comps_cc))
        Sigma_array_masked = np.zeros((num_classes, max_num_comps_cc, num_params, num_params))

        # loop over classes and concatenate means, cov. mats and weights
        for c in range(num_classes):
            
            # initialize arrays
            mu_array = np.zeros((num_params, 1))
            mix_weights = np.zeros(1)
            Sigma_array = np.zeros((1, num_params, num_params))

            # loop over examples in class to construct class conditional
            for n in range(num_pat_per_class_vec[c]):
                mu_array = np.hstack((mu_array, density_estimates[class_idx_mat[n, c]]['mu_array']))
                Sigma_array = np.concatenate((Sigma_array, density_estimates[class_idx_mat[n, c]]['Sigma_array']), axis = 0)
                mix_weights = np.concatenate((mix_weights, density_estimates[class_idx_mat[n, c]]['mix_weights']))
            
            # pop off first entries 
            mu_array = mu_array[:, 1:]
            Sigma_array = Sigma_array[1:, :, :]
            mix_weights = mix_weights[1:]

            # add to masked arrays
            mu_array_masked[c, :, :num_comps_cc_vec[c]] = mu_array
            Sigma_array_masked[c, :num_comps_cc_vec[c], :, :] = Sigma_array
            mix_weights_masked[c, :num_comps_cc_vec[c]] = mix_weights

        # check if Gaussian mixture has spherical cov. matrices
        if self.data_box.get_density_est_opts()['covariance_type'] == 'spherical':
            Sigma_array_masked = Sigma_array_masked[:, :, 0, 0]

        # store everything in dict and return
        masked_arrays_cc = {}

        masked_arrays_cc['labels'] = labels
        masked_arrays_cc['num_comps_vec'] = num_comps_cc_vec

        masked_arrays_cc['mu_array'] = mu_array_masked
        masked_arrays_cc['Sigma_array'] = Sigma_array_masked
        masked_arrays_cc['mix_weights'] = mix_weights_masked

        masked_arrays_cc['num_pat_per_class_vec'] = num_pat_per_class_vec
        masked_arrays_cc['class_idx_mat'] = class_idx_mat


        return masked_arrays_cc

    def _calc_dists_to_class_conds(self, V, density_estimates, class_conditionals):
        '''calculates the distances between the Gaussian mixtures of every example to every class-conditional distribution'''

        # find number of examples and number of classes
        labels = self.data_box.get_labels()
        num_examples = len(density_estimates)
        num_classes = len(class_conditionals)
        
        # find distances
        dist_2_class_cond_mat = np.zeros((num_examples, num_classes))
        dist_2_correct = np.zeros((num_examples, 1)) 
        dist_2_close_incorrect = np.zeros((num_examples, 1)) 
        for n in range(num_examples):
            for c in range(num_classes):
                dist_2_class_cond_mat[n, c] = _dist_L2_Gaussian_mixtures(V,
                                                                        mu_array0 = density_estimates[n]['mu_array'],
                                                                        Sigma_array0 = density_estimates[n]['Sigma_array'], 
                                                                        mix_weights0 = density_estimates[n]['mix_weights'], 
                                                                        mu_array1 = class_conditionals[c]['mu_array'], 
                                                                        Sigma_array1 = class_conditionals[c]['Sigma_array'], 
                                                                        mix_weights1 = class_conditionals[c]['mix_weights'])
            dist_2_correct[n] = dist_2_class_cond_mat[n, int(labels[n])]
            dist_2_close_incorrect[n] = np.min(np.delete(dist_2_class_cond_mat[n, :], int(labels[n])))

        return dist_2_correct, dist_2_close_incorrect

    def _subspace_neg_loglikelihood_and_gradient(self, V_flat, shape_V, class_prior, masked_arrays):
        ''' evaluates the subspace likelihood and calculates the gradient at V '''
        
        # unfold vector VFlat into matrix V
        V = np.copy(V_flat.reshape(shape_V))

        # extract masked arrays
        labels = masked_arrays['labels']
        num_samples_vec = masked_arrays['num_samples_vec']
        num_comps_vec = masked_arrays['num_comps_vec']

        samples_mask = masked_arrays['samples']
        sample_weights_mask = masked_arrays['sample_weights']

        mu_array_mask = masked_arrays['mu_array']
        Sigma_array_mask = masked_arrays['Sigma_array']
        mix_weights_mask = masked_arrays['mix_weights']

        num_pat_per_class_vec = masked_arrays['num_pat_per_class_vec']
        class_idx_mat = masked_arrays['class_idx_mat']

        # check covariance structure
        if self.data_box.get_density_est_opts()['covariance_type'] == 'full':
            # call to NUMBA implemention
            value, f_array, g_array = self._bottleneck_full(V, class_prior, num_samples_vec, num_comps_vec, samples_mask, sample_weights_mask, mu_array_mask, Sigma_array_mask, mix_weights_mask, labels, num_pat_per_class_vec, class_idx_mat)
        elif self.data_box.get_density_est_opts()['covariance_type'] == 'spherical':
            # call to NUMBA implemention
            value, f_array, g_array = self._bottleneck_spherical(V, class_prior, num_samples_vec, num_comps_vec, samples_mask, sample_weights_mask, mu_array_mask, Sigma_array_mask, mix_weights_mask, labels, num_pat_per_class_vec, class_idx_mat)

        # calculate the analytical gradient, pass f_array and g_array to avoid double computation 
        gradient_flat = self._calc_analytical_gradient(V, class_prior, masked_arrays, f_array, g_array)

        # return negative loglikelihood and gradient
        return -value, -gradient_flat

    def _calc_analytical_gradient(self, V, class_prior, masked_arrays, f_array, g_array):
        ''' calculates the gradient at V for the subspace loglikelihood'''
        
        # extract masked arrays
        labels = masked_arrays['labels']
        num_samples_vec = masked_arrays['num_samples_vec']
        num_comps_vec = masked_arrays['num_comps_vec']

        samples_mask = masked_arrays['samples']
        sample_weights_mask = masked_arrays['sample_weights']

        mu_array_mask = masked_arrays['mu_array']
        Sigma_array_mask = masked_arrays['Sigma_array']
        mix_weights_mask = masked_arrays['mix_weights']

        num_pat_per_class_vec = masked_arrays['num_pat_per_class_vec']
        class_idx_mat = masked_arrays['class_idx_mat']
        
        # initialize gradient matrix
        num_rows, num_cols = np.shape(V)
        gradient = np.zeros((num_rows, num_cols))

        # loop over entries of the gradient matrix
        for i in range(num_rows):
            for j in range(num_cols):
                gradient[i, j] = self._calc_analytical_gradient_entry(i, j, V, class_prior, num_samples_vec, num_comps_vec, samples_mask, sample_weights_mask, mu_array_mask, Sigma_array_mask, mix_weights_mask, labels, num_pat_per_class_vec, class_idx_mat, f_array, g_array)
        
        # flatten out gradient to pass to optimizer
        gradient_flat = gradient.flatten()
        
        return gradient_flat

    @staticmethod
    @njit(parallel=True)
    def _calc_analytical_gradient_entry(i, j, V, class_prior, num_samples_vec, num_comps_vec, samples_mask, sample_weights_mask, mu_array_mask, Sigma_array_mask, mix_weights_mask, labels, num_pat_per_class_vec, class_idx_mat, f_array, g_array):
        
        # extact number of examples
        num_examples = len(labels)
        num_classes = len(class_prior)

        # calculate the ambient dimension and wanted subspace dimension
        num_params, num_params_prime = np.shape(V)

        Jij = np.zeros((num_params, num_params_prime))
        Jij[i, j] = 1

        f_prime_array = np.zeros((num_examples, samples_mask.shape[2]))
        g_prime_array = np.zeros((num_examples, samples_mask.shape[2]))

        n_sum_vec = np.zeros(num_examples)
        n_sum_vec_prime = np.zeros(num_examples)

        # loop over examples (to be done in parallel eventually)
        for n in prange(num_examples):

            # loop over samples of example n
            for s in range(num_samples_vec[n]):

                c_sum_vec = np.zeros(num_classes)

                # loop over classes
                for c in range(num_classes):

                    m_k_sum_val = 0.
                    
                    idx_label_c = class_idx_mat[0:num_pat_per_class_vec[c], c]

                    # loop over examples in same class
                    for mm in range(num_pat_per_class_vec[c]):
                        m = idx_label_c[mm]
                        num_mix_comp = num_comps_vec[m]
                        
                        # loop over components of example m
                        for k in range(num_mix_comp):

                            theta = samples_mask[n, :, s]
                            mu = mu_array_mask[m, :, k]
                            Sigma = np.copy(Sigma_array_mask[m, k, :, :])                    # NOTE this could potentially be improved
                            mix_weight = mix_weights_mask[m, k]

                            a = theta - mu
                            Vta = np.dot(np.transpose(V), a)
                            VtSigmaV = np.dot(np.transpose(V), np.dot(Sigma, V))
                            det_VtSigmaV = np.linalg.det(VtSigmaV)
                            V_invVtSigmaV = np.transpose(np.linalg.solve(np.transpose(VtSigmaV), np.transpose(V)))
                            
                            # calculate the determinant term 
                            det_term = ((2 * np.pi) ** (- num_params_prime / 2)) * det_VtSigmaV ** (- 1 / 2)

                            # calculate the derivative of the determinant term 
                            bracket_term = 2 * det_VtSigmaV * np.dot(Sigma, V_invVtSigmaV)
                            dvij_det_term = -0.5 * ((2 * np.pi) ** (- num_params_prime / 2)) * det_VtSigmaV ** (- 3 / 2) * bracket_term[i, j]

                            # calculate the exponential term
                            exp_term = np.exp(- 0.5 * np.dot(np.dot(np.transpose(a), V_invVtSigmaV), Vta))

                            # calculate the derivative of the exponential term
                            Jij_invVtSigmaV_Vt = np.dot(Jij, np.transpose(V_invVtSigmaV))
                            Jijt_Sigma_V = np.dot(np.dot(np.transpose(Jij), Sigma), V)
                            dU_dvij = Jij_invVtSigmaV_Vt - np.dot(np.dot(V_invVtSigmaV, Jijt_Sigma_V + np.transpose(Jijt_Sigma_V)), np.transpose(V_invVtSigmaV)) + np.transpose(Jij_invVtSigmaV_Vt)
                            dvij_quad_term = np.trace(-0.5 * np.dot(np.outer(a, a), dU_dvij))
                            dvij_exp_term = exp_term * dvij_quad_term

                            # calculate the derivative of the multivariate normal PDF
                            dvij_normal = dvij_det_term * exp_term + det_term * dvij_exp_term

                            # add to summation over m and k
                            m_k_sum_val += mix_weight * dvij_normal

                    # add value of weighted mk-sum to the vector for summation over the classes
                    c_sum_vec[c] = m_k_sum_val * (class_prior[c] / num_pat_per_class_vec[c])
                            
                # calculate f_prime and g_prime values
                alpha = sample_weights_mask[n, s]
                f_prime_array[n, s] = alpha * c_sum_vec[labels[n]]
                g_prime_array[n, s] = np.sum(c_sum_vec)
            
            # sum over s
            n_sum_vec[n] = np.sum(f_array[n, :num_samples_vec[n]] / g_array[n, :num_samples_vec[n]])
            n_sum_vec_prime[n] = np.sum((f_prime_array[n, :num_samples_vec[n]] * g_array[n, :num_samples_vec[n]] - f_array[n, :num_samples_vec[n]] * g_prime_array[n, :num_samples_vec[n]]) / (g_array[n, :num_samples_vec[n]] ** 2))
        
        # sum over n
        gradient_entry = np.dot(1 / n_sum_vec, n_sum_vec_prime)

        return gradient_entry

    def _calc_numerical_neg_gradient(self, V_flat, shape_V, class_prior, masked_arrays, epsilon):
        ''' calculates finite difference approximation for gradient of subspace loglikelihood function '''
        
        # unfold vector VFlat into matrix V
        V = np.copy(V_flat.reshape(shape_V))

        # extract masked arrays
        labels = masked_arrays['labels']
        num_samples_vec = masked_arrays['num_samples_vec']
        num_comps_vec = masked_arrays['num_comps_vec']

        samples_mask = masked_arrays['samples']
        sample_weights_mask = masked_arrays['sample_weights']

        mu_array_mask = masked_arrays['mu_array']
        Sigma_array_mask = masked_arrays['Sigma_array']
        mix_weights_mask = masked_arrays['mix_weights']

        num_pat_per_class_vec = masked_arrays['num_pat_per_class_vec']
        class_idx_mat = masked_arrays['class_idx_mat']

        # note gradient is of shape tranpose to V
        gradient = np.zeros(shape_V)

        for i in range(shape_V[0]):
            for j in range(shape_V[1]):
                deviation_mat = np.zeros(shape_V)
                deviation_mat[i, j] = epsilon
                V_plus = V + deviation_mat
                V_minus = V - deviation_mat

                # check covariance structure
                if self.data_box.get_density_est_opts()['covariance_type'] == 'full':
                    # call to NUMBA implemention
                    value_plus, f_array, g_array = self._bottleneck_full(V_plus, class_prior, num_samples_vec, num_comps_vec, samples_mask, sample_weights_mask, mu_array_mask, Sigma_array_mask, mix_weights_mask, labels, num_pat_per_class_vec, class_idx_mat)
                    value_minus, f_array, g_array = self._bottleneck_full(V_minus, class_prior, num_samples_vec, num_comps_vec, samples_mask, sample_weights_mask, mu_array_mask, Sigma_array_mask, mix_weights_mask, labels, num_pat_per_class_vec, class_idx_mat)
                elif self.data_box.get_density_est_opts()['covariance_type'] == 'spherical':
                    # call to NUMBA implemention
                    value_plus, f_array, g_array = self._bottleneck_spherical(V_plus, class_prior, num_samples_vec, num_comps_vec, samples_mask, sample_weights_mask, mu_array_mask, Sigma_array_mask, mix_weights_mask, labels, num_pat_per_class_vec, class_idx_mat)
                    value_minus, f_array, g_array = self._bottleneck_spherical(V_minus, class_prior, num_samples_vec, num_comps_vec, samples_mask, sample_weights_mask, mu_array_mask, Sigma_array_mask, mix_weights_mask, labels, num_pat_per_class_vec, class_idx_mat)

                gradient[i, j] = (value_plus - value_minus) / (2 * epsilon)

        gradient_flat = gradient.flatten()
        
        return -gradient_flat

@njit(parallel=True)
def _calc_dists_to_class_conds_masked_full(V, labels, num_comps_vec, mu_array_mask, Sigma_array_mask, mix_weights_mask, num_pat_per_class_vec, num_comps_vec_cc, mu_array_mask_cc, Sigma_array_mask_cc,  mix_weights_mask_cc):
    '''calculates the distances between the Gaussian mixtures of every example to every class-conditional distribution'''

    # find number of examples and number of classes
    num_examples = len(labels)
    num_classes = len(num_pat_per_class_vec)
    
    # find distances
    dist_2_class_cond_mat = np.zeros((num_examples, num_classes))
    dist_2_correct = np.zeros((num_examples, 1)) 
    dist_2_close_incorrect = np.zeros((num_examples, 1)) 
    for n in prange(num_examples):
        for c in range(num_classes):
            dist_2_class_cond_mat[n, c] = _dist_L2_Gaussian_mixtures(V,
                                                                    mu_array0 = mu_array_mask[n, :, 0:num_comps_vec[n]],
                                                                    Sigma_array0 = Sigma_array_mask[n, 0:num_comps_vec[n], :, :], 
                                                                    mix_weights0 = mix_weights_mask[n, 0:num_comps_vec[n]], 
                                                                    mu_array1 = mu_array_mask_cc[c, :, 0:num_comps_vec_cc[c]],
                                                                    Sigma_array1 = Sigma_array_mask_cc[c, 0:num_comps_vec_cc[c], :, :], 
                                                                    mix_weights1 = mix_weights_mask_cc[c, 0:num_comps_vec_cc[c]])
        dist_2_correct[n] = dist_2_class_cond_mat[n, int(labels[n])]
        dist_2_close_incorrect[n] = np.min(np.delete(dist_2_class_cond_mat[n, :], int(labels[n])))

    return dist_2_correct, dist_2_close_incorrect

@njit
def _dist_L2_Gaussian_mixtures(V, mu_array0, Sigma_array0, mix_weights0, mu_array1, Sigma_array1, mix_weights1):
    ''' calculates the L2-norm between the Gaussian mixtures given by (mu_array0, Sigma_array0, mix_weights0) and (mu_array1, Sigma_array1, mix_weights1) when projected onto the columnspace of V'''
    
    # extract number of components
    num_comps0 = len(mix_weights0)
    num_comps1 = len(mix_weights1)

    # calculate quadratic term for mixture 0
    quad_term0 = 0.
    for i in range(num_comps0):
        for j in range(num_comps0):

            x = mu_array0[:, i]
            mu = mu_array0[:, j]
            Sigma = Sigma_array0[i, :, :] + Sigma_array0[j, :, :]

            a = x - mu
            VtSigmaV = np.dot(np.dot(np.transpose(V), Sigma), V)
            detVtSigmaV = np.linalg.det(VtSigmaV)
            Vta = np.dot(np.transpose(V), a)
            invVtSigmaVVta = np.linalg.solve(VtSigmaV, Vta)
            mvnpdfEval = (2 * np.pi) ** (-V.shape[1] / 2) * detVtSigmaV ** (-0.5) * np.exp(-0.5 * np.dot(np.transpose(Vta), invVtSigmaVVta))
            quad_term0 += mix_weights0[i] * mix_weights0[j] * mvnpdfEval

            #quad_term0 += mix_weights0[i] * mix_weights0[j] * multivariate_normal.pdf(mu_array0[:, i], mean = mu_array0[:, j], cov = Sigma_array0[i, :, :] + Sigma_array0[j, :, :])

    # calculate mix-term between mixtures 0 and 1
    mix_term = 0.
    for i in range(num_comps0):
        for j in range(num_comps1):

            x = mu_array0[:, i]
            mu = mu_array1[:, j]
            Sigma = Sigma_array0[i, :, :] + Sigma_array1[j, :, :]

            a = x - mu
            VtSigmaV = np.dot(np.dot(np.transpose(V), Sigma), V)
            detVtSigmaV = np.linalg.det(VtSigmaV)
            Vta = np.dot(np.transpose(V), a)
            invVtSigmaVVta = np.linalg.solve(VtSigmaV, Vta)
            mvnpdfEval = (2 * np.pi) ** (-V.shape[1] / 2) * detVtSigmaV ** (-0.5) * np.exp(-0.5 * np.dot(np.transpose(Vta), invVtSigmaVVta))
            mix_term += mix_weights0[i] * mix_weights1[j] * mvnpdfEval

            #mix_term += mix_weights0[i] * mix_weights1[j] * multivariate_normal.pdf(mu_array0[:, i], mean = mu_array1[:, j], cov = Sigma_array0[i, :, :] + Sigma_array1[j, :, :])

    # calculate quadratic term for mixture 1
    quad_term1 = 0.
    for i in range(num_comps1):
        for j in range(num_comps1):

            x = mu_array1[:, i]
            mu = mu_array1[:, j]
            Sigma = Sigma_array1[i, :, :] + Sigma_array1[j, :, :]

            a = x - mu
            VtSigmaV = np.dot(np.dot(np.transpose(V), Sigma), V)
            detVtSigmaV = np.linalg.det(VtSigmaV)
            Vta = np.dot(np.transpose(V), a)
            invVtSigmaVVta = np.linalg.solve(VtSigmaV, Vta)
            mvnpdfEval = (2 * np.pi) ** (-V.shape[1] / 2) * detVtSigmaV ** (-0.5) * np.exp(-0.5 * np.dot(np.transpose(Vta), invVtSigmaVVta))
            quad_term1 += mix_weights1[i] * mix_weights1[j] * mvnpdfEval

            #quad_term1 += mix_weights1[i] * mix_weights1[j] * multivariate_normal.pdf(mu_array1[:, i], mean = mu_array1[:, j], cov = Sigma_array1[i, :, :] + Sigma_array1[j, :, :])

    dist = quad_term0 - 2 * mix_term + quad_term1

    return dist

@njit(parallel=True)
def _calc_dists_to_class_conds_masked_spherical(V, labels, num_comps_vec, mu_array_mask, Sigma_array_mask, mix_weights_mask, num_pat_per_class_vec, num_comps_vec_cc, mu_array_mask_cc, Sigma_array_mask_cc,  mix_weights_mask_cc):
    '''calculates the distances between the Gaussian mixtures of every example to every class-conditional distribution'''

    # find number of examples and number of classes
    num_examples = len(labels)
    num_classes = len(num_pat_per_class_vec)
    
    # find distances
    dist_2_class_cond_mat = np.zeros((num_examples, num_classes))
    dist_2_correct = np.zeros((num_examples, 1)) 
    dist_2_close_incorrect = np.zeros((num_examples, 1)) 
    for n in prange(num_examples):
        for c in range(num_classes):
            dist_2_class_cond_mat[n, c] = _dist_L2_Gaussian_mixtures_spherical(V,
                                                                    mu_array0 = mu_array_mask[n, :, 0:num_comps_vec[n]],
                                                                    Sigma_array0 = Sigma_array_mask[n, 0:num_comps_vec[n]], 
                                                                    mix_weights0 = mix_weights_mask[n, 0:num_comps_vec[n]], 
                                                                    mu_array1 = mu_array_mask_cc[c, :, 0:num_comps_vec_cc[c]],
                                                                    Sigma_array1 = Sigma_array_mask_cc[c, 0:num_comps_vec_cc[c]], 
                                                                    mix_weights1 = mix_weights_mask_cc[c, 0:num_comps_vec_cc[c]])
        dist_2_correct[n] = dist_2_class_cond_mat[n, int(labels[n])]
        dist_2_close_incorrect[n] = np.min(np.delete(dist_2_class_cond_mat[n, :], int(labels[n])))

    return dist_2_correct, dist_2_close_incorrect

@njit
def _dist_L2_Gaussian_mixtures_spherical(V, mu_array0, Sigma_array0, mix_weights0, mu_array1, Sigma_array1, mix_weights1):
    ''' calculates the L2-norm between the Gaussian mixtures given by (mu_array0, Sigma_array0, mix_weights0) and (mu_array1, Sigma_array1, mix_weights1) when projected onto the columnspace of V'''
    
    # NOTE this function can be made faster by a factor of 2 by using symmetry of the Gaussian

    # extract number of components
    num_comps0 = len(mix_weights0)
    num_comps1 = len(mix_weights1)

    # find determinant and inverse of V^T * V
    det_VtV = np.linalg.det(np.dot(np.transpose(V), V))
    inv_VtV = np.linalg.inv(np.dot(np.transpose(V), V))
    num_col_V = np.shape(V)[1]

    # calculate quadratic term for mixture 0
    quad_term0 = 0.
    for i in range(num_comps0):
        for j in range(num_comps0):

            x = mu_array0[:, i]
            mu = mu_array0[:, j]
            sigma_sq = Sigma_array0[i] + Sigma_array0[j]

            a = x - mu
            detVtSigmaV = (sigma_sq ** num_col_V) * det_VtV
            Vta = np.dot(np.transpose(V), a)
            invVtSigmaVVta = np.dot((1 / sigma_sq) * inv_VtV, Vta)
            mvnpdfEval = (2 * np.pi) ** (-num_col_V / 2) * detVtSigmaV ** (-0.5) * np.exp(-0.5 * np.dot(np.transpose(Vta), invVtSigmaVVta))
            quad_term0 += mix_weights0[i] * mix_weights0[j] * mvnpdfEval

            #quad_term0 += mix_weights0[i] * mix_weights0[j] * multivariate_normal.pdf(mu_array0[:, i], mean = mu_array0[:, j], cov = Sigma_array0[i, :, :] + Sigma_array0[j, :, :])

    # calculate mix-term between mixtures 0 and 1
    mix_term = 0.
    for i in range(num_comps0):
        for j in range(num_comps1):

            x = mu_array0[:, i]
            mu = mu_array1[:, j]
            sigma_sq = Sigma_array0[i] + Sigma_array1[j]

            a = x - mu
            detVtSigmaV = (sigma_sq ** num_col_V) * det_VtV
            Vta = np.dot(np.transpose(V), a)
            invVtSigmaVVta = np.dot((1 / sigma_sq) * inv_VtV, Vta)
            mvnpdfEval = (2 * np.pi) ** (-num_col_V / 2) * detVtSigmaV ** (-0.5) * np.exp(-0.5 * np.dot(np.transpose(Vta), invVtSigmaVVta))
            mix_term += mix_weights0[i] * mix_weights1[j] * mvnpdfEval

            #mix_term += mix_weights0[i] * mix_weights1[j] * multivariate_normal.pdf(mu_array0[:, i], mean = mu_array1[:, j], cov = Sigma_array0[i, :, :] + Sigma_array1[j, :, :])

    # calculate quadratic term for mixture 1
    quad_term1 = 0.
    for i in range(num_comps1):
        for j in range(num_comps1):

            x = mu_array1[:, i]
            mu = mu_array1[:, j]
            sigma_sq = Sigma_array1[i] + Sigma_array1[j]

            a = x - mu
            detVtSigmaV = (sigma_sq ** num_col_V) * det_VtV
            Vta = np.dot(np.transpose(V), a)
            invVtSigmaVVta = np.dot((1 / sigma_sq) * inv_VtV, Vta)
            mvnpdfEval = (2 * np.pi) ** (-num_col_V / 2) * detVtSigmaV ** (-0.5) * np.exp(-0.5 * np.dot(np.transpose(Vta), invVtSigmaVVta))
            quad_term1 += mix_weights1[i] * mix_weights1[j] * mvnpdfEval

            #quad_term1 += mix_weights1[i] * mix_weights1[j] * multivariate_normal.pdf(mu_array1[:, i], mean = mu_array1[:, j], cov = Sigma_array1[i, :, :] + Sigma_array1[j, :, :])

    dist = quad_term0 - 2 * mix_term + quad_term1

    return dist