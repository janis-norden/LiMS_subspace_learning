from ..dynamical_system import DynamicalSystem
from ..data_box import DataBox
import numpy as np
import scipy

class Spiral(DynamicalSystem):

    ### attributes ###
    name = 'spiral'                                         # string with name of the system
    ROI = np.array([[0, 0.1],                               # matrix defining region of interest
                    [0, 0.1],
                    [0, 0.1]])                                      
    num_states = None                                       # number of state variables in the model
    num_params = 3                                          # number of parameters in the model

    ### methods: polymorphisms ###

    def __init__(self):
        super().__init__()

    ### methods: model-specific ###
    def gen_artificial_data(self, gen_data_opts):

        ''' generate artificial data for the 3D spiral classes '''

        # number of examples
        num_examples = gen_data_opts['num_examples']
        num_samples = gen_data_opts['num_samples']

        # 2D covariance matrix associated with the ground truth class-distribution
        shift_vec = gen_data_opts['shift_vec']
        scale_spiral = gen_data_opts['scale_spiral']
        tangent_stretch  = gen_data_opts['tangent_stretch']
        num_intervals = gen_data_opts['num_intervals']
        num_comps = gen_data_opts['num_comps']
        scale_cov = gen_data_opts['scale_cov']
        z_std = gen_data_opts['z_std']
        diag_project = gen_data_opts['diag_project']

        # determine center point and basis vectors for subspace
        subspace = {}
        subspace['center'] = np.concatenate(([0], shift_vec))
        subspace['v1'] = np.array([1, 0, 0])
        subspace['v2'] = np.array([0, 1, 0])

        # initialize
        labels = np.zeros(np.sum(num_examples))

        # set parameters for spiral regularity
        s0 = 3
        s1 = 10
        delta = (s1 - s0) / num_intervals

        # initialize dict
        density_est_dict = {}
        posteriors_dict = {}

        cnt = 0
        for class_idx in range(len(num_examples)):
            for n in range(num_examples[class_idx]):
                
                # initialize mean, cov mat and mix weights arrays
                mu_array = np.zeros((3, num_comps))
                Sigma_array = np.zeros((num_comps, 3, 3))
                mix_weights = np.ones(num_comps) / num_comps
                samples = []

                for k in range(num_comps):

                    # determine active interval to draw from and draw point as mean location
                    interval = np.mod(n, num_intervals)
                    s = np.random.uniform(s0 + interval * delta, s0 + (interval + 1) * delta, 1) 

                    # separate by classes
                    if class_idx == 0:
                        mu_2D = scale_spiral * s * np.concatenate(( np.sin(s), np.cos(s))) + shift_vec
                        tangent_2D = scale_spiral * np.concatenate(( np.sin(s) + s * np.cos(s), np.cos(s) - s * np.sin(s)))
                        tangent_2D = tangent_2D / np.linalg.norm(tangent_2D)
                        tangent = np.vstack((np.reshape(tangent_2D, (2, 1)), 0))

                    else:
                        mu_2D = - scale_spiral * s * np.concatenate((np.sin(s), np.cos(s))) + shift_vec
                        tangent_2D = - scale_spiral * np.concatenate(( np.sin(s) + s * np.cos(s), np.cos(s) - s * np.sin(s)))
                        tangent_2D = tangent_2D / np.linalg.norm(tangent_2D)
                        tangent = np.vstack((np.reshape(tangent_2D, (2, 1)), 0))
                    
                    # create mu
                    z_comp = np.random.normal(loc = 0, scale = z_std)
                    mu = np.append(mu_2D, z_comp)

                    # create covariance matrix
                    normal_vec = np.array([[0.], [0.], [1.]])
                    binormal_vec = np.cross(tangent[:, 0], normal_vec[:, 0])
                    T = np.hstack((tangent, normal_vec, np.reshape(binormal_vec, (3, 1))))
                    cov_mat = np.dot(np.dot(T, scale_cov * np.diag([tangent_stretch, 1 / np.sqrt(tangent_stretch), 1 / np.sqrt(tangent_stretch)])), np.linalg.inv(T))

                    # only keep diagonal terms if diag_project is True
                    if diag_project:
                        cov_mat = np.diag(np.diag(cov_mat))

                    # create "posterior" samples
                    samples.append(np.random.multivariate_normal(mu, cov_mat, num_samples))

                    # add to mixture arrays
                    mu_array[:, k] = mu
                    Sigma_array[k, :, :] = cov_mat

                # create dict to hold means, covariances and mixture weights
                density_estimate = {}
                density_estimate['mu_array'] = mu_array
                density_estimate['Sigma_array'] = Sigma_array
                density_estimate['mix_weights'] = mix_weights

                # convert list of samples to single array, set equal weights
                samples = np.concatenate(samples)
                weights = np.ones(samples.shape[0]) / samples.shape[0]

                # add posterior samples to dict
                posterior = {}
                posterior['samples'] = samples
                posterior['weights'] = weights

                # add to dict holding the density estimates
                density_est_dict[cnt] = density_estimate
                posteriors_dict[cnt] = posterior

                # update 
                labels[cnt] = class_idx
                cnt += 1

            # create DataBox and add data to it
            data_box = DataBox()

            data_box.set_labels(labels)
            data_box.set_density_estimates(density_est_dict)
            data_box.set_posterior_samples(posteriors_dict)

            data_box.set_dynamical_system(self)
            data_box.set_subspace(subspace)
            data_box.set_gen_data_opts(gen_data_opts)

        return data_box