import numpy as np
import scipy
from copy import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_masked_arrays(data_box):
    ''' extract data from data box and collect everything into masked np.arrays '''

    # extract from data box
    covariance_type = data_box.get_density_est_opts()['covariance_type']
    labels = data_box.get_labels().astype(int)
    posterior_samples = data_box.get_posterior_samples()
    density_estimates = data_box.get_density_estimates()
    num_params = data_box.get_dynamical_system().num_params

    # determine the total number of examples
    num_examples = len(labels)
    
    # determine max number of samples and components
    num_samples_vec = np.zeros(num_examples, dtype=int)
    num_comps_vec = np.zeros(num_examples, dtype=int)
    for i in range(num_examples):
        num_samples_vec[i] = len(posterior_samples[i]['weights'])
        num_comps_vec[i] = len(density_estimates[i]['mix_weights'])
    max_num_samples = np.max(num_samples_vec)
    max_num_comps = np.max(num_comps_vec)

    # initialize arrays with max dimensions
    samples_mask = np.zeros((num_examples, num_params, max_num_samples))
    sample_weights_mask = np.zeros((num_examples, max_num_samples))
    mu_array_mask = np.zeros((num_examples, num_params, max_num_comps))
    mix_weights_mask = np.zeros((num_examples, max_num_comps))

    # check covariance structure
    if covariance_type == 'full':
        Sigma_array_mask = np.zeros((num_examples, max_num_comps, num_params, num_params))
    elif covariance_type == 'spherical':
        Sigma_array_mask = np.zeros((num_examples, max_num_comps))

    # fill in arrays only up to the corresponding number of samples / mixture components
    for i in range(num_examples):
        samples_mask[i, :, 0:num_samples_vec[i]] = posterior_samples[i]['samples'].T
        sample_weights_mask[i, 0:num_samples_vec[i]] = posterior_samples[i]['weights']
        
        mu_array_mask[i, :, 0:num_comps_vec[i]] = density_estimates[i]['mu_array'] 
        mix_weights_mask[i, 0:num_comps_vec[i]] = density_estimates[i]['mix_weights']

        if covariance_type == 'full':
            Sigma_array_mask[i, 0:num_comps_vec[i], :, :] = density_estimates[i]['Sigma_array']
        elif covariance_type == 'spherical':
            Sigma_array_mask[i, 0:num_comps_vec[i]] = density_estimates[i]['Sigma_array'][:, 0, 0]
        
    # determine how many patients there are per class and the maximum number of patients in a single class
    unique, counts = np.unique(labels, return_counts=True)
    num_pat_per_class_vec = counts
    num_classes = len(unique)
    
    # put patient indices into single matrix
    max_pat_in_class = max(num_pat_per_class_vec)
    class_idx_mat = np.zeros((max_pat_in_class, num_classes), dtype=int)
    for c in range(num_classes):
        class_idx_mat[0:num_pat_per_class_vec[c], c] = [idx for idx, value in enumerate(labels) if value == c]

    # store everything in dict and return
    masked_arrays = {}

    masked_arrays['labels'] = labels
    masked_arrays['num_samples_vec'] = num_samples_vec
    masked_arrays['num_comps_vec'] = num_comps_vec

    masked_arrays['samples'] = samples_mask
    masked_arrays['sample_weights'] = sample_weights_mask

    masked_arrays['mu_array'] = mu_array_mask
    masked_arrays['Sigma_array'] = Sigma_array_mask
    masked_arrays['mix_weights'] = mix_weights_mask

    masked_arrays['num_pat_per_class_vec'] = num_pat_per_class_vec
    masked_arrays['class_idx_mat'] = class_idx_mat

    return masked_arrays

def set_class_prior(labels, prior_type):
    ''' returns np.array containing the discrete class prior probabilities '''
    
    # check which class prior type is wanted
    if prior_type == 'empirical':
        unique, counts = np.unique(labels, return_counts=True)
        class_prior = counts / len(labels)
    elif prior_type == 'flat':
        unique = np.unique(labels)
        class_prior = np.ones(len(unique)) / len(unique)

    return class_prior

def calc_accuracy(predictions, labels, winner_takes_all = True):
    '''calculates the accuracy given a matrix of class predictions and the label vector
    
    predictions:                 numpy array, (num_examples, num_classes)
    labels:                      numpy array, (num_examples,)
    winner_takes_all:            boolean, indicates whether to use winner takes all scheme for predictions or probabilities

    '''

    # calculate confusion matrix
    conf_mat = calc_confusion_matrix(predictions, labels, winner_takes_all = winner_takes_all)
        
    # find micro averaged accuracy
    diag_sum = np.sum(np.diag(conf_mat))
    off_diag_sum = np.sum(conf_mat) - diag_sum
    micro_acc = diag_sum / (diag_sum + off_diag_sum)

    # find macro averaged accuracy
    row_sums= np.sum(conf_mat, axis=1)
    class_avg = np.diag(conf_mat) / row_sums
    macro_acc = np.mean(class_avg)
    
    return micro_acc, macro_acc

def calc_log_likelihood(predictions, labels):

    '''calculates the log-likelihood given a matrix of class predictions and the associated label vector
    
    predictions:                 numpy array, (num_examples, num_classes)
    labels:                      numpy array, (num_examples,)

    '''

    num_examples = len(labels)
    log_likelihood = 0
    for n in range(num_examples):
            log_likelihood += np.log(predictions[n, int(labels[n])])

    return log_likelihood

def calc_BIC(num_parameters, num_observations, log_likelihood):
    '''calculates the Bayesian Information Criterion (BIC)
    
    num_parameters:                 int
    num_observations:               int
    likelihood:                     float

    '''

    return num_parameters * np.log(num_observations) - 2 * log_likelihood

def calc_AIC(num_parameters, log_likelihood):
    '''calculates the Akaike Information Criterion (AIC)
    
    num_parameters:                 int
    likelihood:                     float

    '''

    return 2 * num_parameters  - 2 * log_likelihood

def calc_confusion_matrix(predictions, labels, winner_takes_all = True):
    '''calculates the confusion matrix given a matrix of class predictions and the label vector
        
        predictions:                 numpy array, (num_examples, num_classes)
        labels:                      numpy array, (num_examples,)
        winner_takes_all:            boolean, indicates whether to use winner takes all scheme for predictions or probabilities

        '''
    # initialize confusion matrix
    labels = labels.astype(int)
    values, counts = np.unique(labels, return_counts=True)
    num_examples, num_classes = np.shape(predictions)
    conf_mat = np.zeros((num_classes, num_classes))
    
    if winner_takes_all:
        for n in range(num_examples):
            indicator_row = np.zeros(num_classes)
            indicator_row[np.argmax(predictions[n, :])] = 1
            old_row = np.copy(conf_mat[labels[n], :])
            #conf_mat[labels[n], :] = old_row + (1 / counts[labels[n]]) * indicator_row
            conf_mat[labels[n], :] = old_row + indicator_row
    else:
        for n in range(num_examples):
            old_row = np.copy(conf_mat[labels[n], :])
            #conf_mat[labels[n], :] = old_row + (1 / counts[labels[n]]) * predictions[n, :]
            conf_mat[labels[n], :] = old_row +  predictions[n, :]
    
    return conf_mat

def plot_spirals(data_box, plot_opts):
    '''creates 3D plot of binary-classification problem for the spiral model'''

    # extrat from data
    density_estimates = data_box.get_density_estimates()
    labels = data_box.get_labels()

    # extract plot options
    x_lim = plot_opts['x_lim']
    y_lim = plot_opts['y_lim']
    z_lim = plot_opts['z_lim']

    x_label = plot_opts['x_label']
    y_label = plot_opts['y_label']
    z_label = plot_opts['z_label']

    # create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    # 95% confidence interval
    nstd = 3    

    # loop over examples and plot covariance ellipsoids
    for i in range(len(density_estimates)):
        for k in range(len(density_estimates[i]['mix_weights'])):
            mu = density_estimates[i]['mu_array'][:, k]
            cov = density_estimates[i]['Sigma_array'][k, :, :]
            X1,Y1,Z1 = _get_cov_ellipsoid(cov, mu, nstd)
            if labels[i] == 0:
                ax.plot_wireframe(X1,Y1,Z1, color='b', alpha=0.1)
            else:
                ax.plot_wireframe(X1,Y1,Z1, color='r', alpha=0.1)
    ax.plot(0, 0, 'b', label = 'class 0')
    ax.plot(0, 0, 'r', label = 'class 1')
    ax.legend()
    plt.show()
    
    return ax

def _get_cov_ellipsoid(cov, mu=np.zeros((3)), nstd=3):
    """
    Return the 3d points representing the covariance matrix
    cov centred at mu and scaled by the factor nstd.

    Plot on your favourite 3d axis. 
    Example 1:  ax.plot_wireframe(X,Y,Z,alpha=0.1)
    Example 2:  ax.plot_surface(X,Y,Z,alpha=0.1)

    Source: https://github.com/CircusMonkey/covariance-ellipsoid/blob/master/ellipsoid.py
    """
    assert cov.shape==(3,3)

    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.sum(cov,axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    # Set of all spherical angles to draw our ellipsoid
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Width, height and depth of ellipsoid
    rx, ry, rz = nstd * np.sqrt(eigvals)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off axis alignment
    old_shape = X.shape
    # Flatten to vectorise rotation
    X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
    X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
    X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)

    # Add in offsets for the mean
    X = X + mu[0]
    Y = Y + mu[1]
    Z = Z + mu[2]
    
    return X,Y,Z

def GrassmannDist(A, B):
    '''
    calculates the Grassmann distance between the subapces spanned by the columns of A and B
    '''

    # find orthonormal bases for range(A) and range(B)
    orth_norm_base_A = scipy.linalg.orth(A)
    orth_norm_base_B = scipy.linalg.orth(B)

    # compute SVD of A'^T * B'
    U, S, Vh = np.linalg.svd(np.dot(np.transpose(orth_norm_base_A), orth_norm_base_B))

    # determine the principle angles between the subspaces
    princ_angles = np.arccos(np.clip(S, -1, 1))

    # determine Grassmann distance
    grass_dist = np.linalg.norm(princ_angles)

    return grass_dist