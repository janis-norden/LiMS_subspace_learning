import numpy as np
import sys, os, pickle
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
import corner
from copy import deepcopy

class DataBox:

    # per observation attributes
    _labels = None               # np array of class labels
    _init_conds = None           # np array of initial conditions
    _parameters = None           # np array of parameters
    _timeseries = None           # dict of time series data
    _posterior_samples = None    # dict containing np array 'samples' of posterior samples and np array 'weights' of sample weights
    _density_estimates = None    # dict containing np arrays 'mu_array', 'Sigma_array' and 'mix_weights' contianing the means, covariances and mixture weights of the GM model

    # data generation and processing attributes
    _dynamical_system = None     # object of type dynamical system
    _subspace = None             # dict containing info about the true subspace
    _subspace_learned = None     # to be determined
    _gen_data_opts = None        # dict containing the data generation options
    _sampler_opts = None         # dict containing the sampler options
    _density_est_opts = None     # dict containing the density estimation options
    _subspace_learn_opts = None  # dict containing the subspace learning options

    def __init__(self):
        # allows data box to be initialized without having set any attributes
        
        pass  

    # get and set methods

    def get_labels(self):
        return self._labels
    
    def set_labels(self, value):
        self._labels = value

    def get_init_conds(self):
        return self._init_conds
    
    def set_init_conds(self, value):
        self._init_conds = value

    def get_parameters(self):
        return self._parameters
    
    def set_parameters(self, value):
        self._parameters = value
    
    def get_timeseries(self):
        return self._timeseries
    
    def set_timeseries(self, value):
        self._timeseries = value

    def get_posterior_samples(self):
        return self._posterior_samples
    
    def set_posterior_samples(self, value):
        self._posterior_samples = value

    def get_density_estimates(self):
        return self._density_estimates
    
    def set_density_estimates(self, value):
        self._density_estimates = value

    def get_subspace_learned(self):
        return self._subspace_learned
    
    def set_subspace_learned(self, value):
        self._subspace_learned = value




    def get_dynamical_system(self):
        return self._dynamical_system
    
    def set_dynamical_system(self, value):
        self._dynamical_system = value

    def get_subspace(self):
        return self._subspace
    
    def set_subspace(self, value):
        self._subspace = value

    def get_gen_data_opts(self):
        return self._gen_data_opts
    
    def set_gen_data_opts(self, value):
        self._gen_data_opts = value

    def get_sampler_opts(self):
        return self._sampler_opts
    
    def set_sampler_opts(self, value):
        self._sampler_opts = value

    def get_density_est_opts(self):
        return self._density_est_opts
    
    def set_density_est_opts(self, value):
        self._density_est_opts = value

    def get_subspace_learn_opts(self):
        return self._subspace_learn_opts
    
    def set_subspace_learn_opts(self, value):
        self._subspace_learn_opts = value


    # other methods
    def save_data(self, file_name = None):
        # save current state of data_box, 
        # files will be saved to the data/interim folder
        # if no filename is provided, the filename is set to be the name of the dynamical system and the current date and time are attached to give a unique name

        # find path to project root
        path_to_project_root = str(Path(__file__).parents[1])

        # check if filename has been provided, if not, set file name tail to be the current date and time
        if file_name == None:

            # find current time and date
            current_date_time = datetime.now()
            date_time_str = current_date_time.strftime("%Y_%m_%d_%H_%M_%S")

            # determine path to Python/data/
            save_path = path_to_project_root + '/data/interim/' + self._dynamical_system.name + '_' + date_time_str + '.pckl'

        else:
            # determine path to Python/data/
            save_path = path_to_project_root + '/data/interim/' + self._dynamical_system.name + '_' + file_name + '.pckl'
        
        # open a file, save data and close
        f = open(save_path, 'wb')
        pickle.dump(self, f)
        f.close()

    def load_data(self, path_to_file):
        # load state of data_box from file, the string path_to_file is the path to the file to be loaded relative to Python/data

        # find path to project root
        path_to_project_root = str(Path(__file__).parents[1])

        # append path to file
        load_path = path_to_project_root + '/data' + path_to_file

        # load file
        f = open(load_path, 'rb')
        data_box = pickle.load(f)
        f.close()

        return data_box
    
    def _merge_samples_by_class(self):
        
        """ merge samples from all examples belonging to the same class and return dict containing the collected samples"""

        # extract from self
        labels = self._labels
        posterior_samples = self._posterior_samples

        # find number of classes and class memberships, loop over classes
        classes = np.unique(labels)
        class_samples = {}

        # loop over classes
        for idx, c in enumerate(classes):

            # find indices of all examples belonging to class c
            class_indices = [i for i, label in enumerate(labels) if label == c]

            # merge all samples of class c and store in class_samples
            samples_all = posterior_samples[class_indices[0]]['samples']
            for i in class_indices[1:]:
                samples_all = np.vstack((samples_all, posterior_samples[i]['samples']))
            class_samples[idx] = samples_all

        return class_samples

    def merge_samples_all(self):

        """ merge samples from all examples and return samples as np array"""

        # extract from self
        labels = self._labels
        posterior_samples = self._posterior_samples

        # loop over examples
        samples_all = posterior_samples[0]['samples']
        for i in range(len(posterior_samples) - 1):
            samples_all = np.vstack((samples_all, posterior_samples[i + 1]['samples']))
            
        return samples_all

    def merge_means_and_labels(self):

        '''
        Merge means and labels.

        Given a data box for which density estimates have already been computed, 
        this function merges together all mean vectors of all components in each mixture into a single array.
        Additionally, a new label array is constructed s.t. the mean vector of each component is assigned the 
        label of the Gaussian mixture.

        Parameters
        ----------
        
        Returns
        -------

        means_merged    :   (sum_{n=1}^{num_examples} num_components(n), num_features) array_like
                            Numpy array containing all mean vectors
        
        labels_merged   :   (sum_{n=1}^{num_examples} num_components(n), ) array_like
                            Numpy array containing all labels
        '''

        # extract labels and density estimates from self
        labels = self.get_labels()
        density_estimates = self.get_density_estimates()

        # extract number of examples
        num_examples = len(labels)

        # loop over all examples and concatenate mean vectors in single array, collect labels
        means_list = []
        labels_list = []
        for n in range(num_examples):

            # find number of components
            num_components = len(density_estimates[n]['mix_weights'])

            # add the correct label a number of time to the labels list
            labels_list.append(labels[n] * np.ones(num_components))

            # add means to means list
            means_list.append(density_estimates[n]['mu_array'].T)

        # merge list to arrays
        labels_merged = np.concatenate(labels_list)
        means_merged = np.concatenate(means_list)

        return means_merged, labels_merged

    def merge_classes(self, class_idx):
        ''' given a data box, it replaces all labels that are contained in class_idx with the minimum element of class_idx, 
        it then recalculates the labels to be increasing integers from 0 to c, thus mergeing the classes indicated in class_idx '''

        # make copy of data box
        data_box_merged = deepcopy(self)
        labels = data_box_merged.get_labels()
        labels_replaced = np.copy(labels)

        # find minimum label and set as new label
        new_label = np.min(class_idx)

        # loop over examples and set all labels in class_idx equal to new_label
        num_examples = len(labels)
        for n in range(num_examples):
            if labels[n] in class_idx:
                labels_replaced[n] = new_label

        # find new unique values in labels
        values = np.unique(labels_replaced)

        # assign new labels from 0 to c
        labels_merged = np.copy(labels_replaced)
        for c in range(len(values)):
            idx_c  = np.where(labels_replaced == values[c])
            labels_merged[idx_c] = c

        data_box_merged.set_labels(labels_merged)

        return data_box_merged

    def down_sample_examples(self, class_idx, return_amounts):
        ''' given a data box, down sample examples from classes specified in class_idx to the amounts specified in return_amounts'''    # NOTE this function currently only applies to posterior samples

        # make copy of data box
        data_box_down_sampled = deepcopy(self)
        labels = data_box_down_sampled.get_labels()
        
        # find unique classes and the amount of examples per class
        values, counts = np.unique(labels, return_counts=True)

        # loop over classes
        idx_remain_list = []
        for c in values:

            # find class indices
            idx_c = np.argwhere(labels == c)[:, 0]

            # check if current class is mentioned in class_idx, if so, down-sample to amount specified in return_amounts
            if c in class_idx:
                return_amount = return_amounts[class_idx == c][0]
            else:
                return_amount = counts[c]

            # randomly sample return_amount examples from class c
            idx_c_remain = np.random.choice(idx_c, return_amount, replace=False)
            idx_remain_list.append(idx_c_remain)

        # concatenate all indices of examples remaining
        idx_remain = np.concatenate(idx_remain_list)

        # create new data box that only contains examples mentioned in idx_remain
        data_box_down_sampled = self.select_examples(idx_remain)

        return data_box_down_sampled

    def select_examples(self, selection_indices):
        '''Return new databox that is identical to self but only has observations specified in selection_indices'''

        # make copy of data box
        data_box_new = deepcopy(self)

        labels = data_box_new.get_labels()
        init_conds = data_box_new.get_init_conds()
        parameters = data_box_new.get_parameters()
        timeseries = data_box_new.get_timeseries()
        posterior_samples = data_box_new.get_posterior_samples()
        density_estimates = data_box_new.get_density_estimates()

        if labels is not None:
            labels_new = labels[selection_indices]
            data_box_new.set_labels(labels_new)

        if init_conds is not None:
            init_conds_new = init_conds[selection_indices, :]
            data_box_new.set_init_conds(init_conds_new) 

        if parameters is not  None:
            parameters_new = parameters[selection_indices, :]
            data_box_new.set_parameters(parameters_new)
        
        if timeseries is not  None:
            timeseries_new = {}
            for n in range(len(selection_indices)):
                timeseries_new[n] = timeseries[selection_indices[n]]
            data_box_new.set_timeseries(timeseries_new)

        if posterior_samples is not None:
            posterior_samples_new = {}
            for n in range(len(selection_indices)):
                posterior_samples_new[n] = posterior_samples[selection_indices[n]]
            data_box_new.set_posterior_samples(posterior_samples_new)

        if density_estimates is not None:
            density_estimates_new = {}
            for n in range(len(selection_indices)):
                density_estimates_new[n] = density_estimates[selection_indices[n]]
            data_box_new.set_density_estimates(density_estimates_new)

        return data_box_new

    def shift_and_scale_data(self, shifts = None, scalings = None):
        
        '''
        Standardize posterior samples and density estimates.

        Applies the standard-score transformation to the samples of each feature, i.e. Z = (X - mu) / sigma.
        If no arrays containing means and standard deviations are provided, 
        the means and standard deviations of each feature are computed based on the data contained in self.

        Parameters
        ----------
        shifts   :      (num_features, ) array_like
                        array containing the values by which to shift each feature
        scalings :      (num_features, ) array_like
                        array containing the values by which to scale the shifted features
        
        Returns
        -------

        data_box_new    :   DataBox object 
                            Data box that now contains the transformed data
            
        '''

        # create new data box
        data_box_new = deepcopy(self)

        # extract posterior samples and density estimates
        posterior_samples = data_box_new.get_posterior_samples()
        density_estimates = data_box_new.get_density_estimates()
        num_examples = len(posterior_samples)

        # check if normalization is based on givend data (i.e. means or st_devs is None) or external reference
        if shifts is None or scalings is None:
            
            # merge samples and get means and std. for each feature
            samples_merged = data_box_new.merge_samples_all()

            # create output arrays as mean and std of posterior samples
            shifts = np.mean(samples_merged, axis=0)
            scalings = np.std(samples_merged, axis=0)

        # define scaling matrix D
        D = np.diag(1 / scalings)

        # loop over examples
        for n in range(num_examples):

            # transform posterior samples
            num_samples = len(posterior_samples[n]['weights'])
            means_stacked_s = np.repeat(shifts[np.newaxis], repeats = num_samples, axis=0)
            posterior_samples[n]['samples'] = np.dot(posterior_samples[n]['samples'] - means_stacked_s, D)

            # transform density estimates
            num_components = len(density_estimates[n]['mix_weights'])
            means_stacked_d = np.repeat(shifts[np.newaxis], repeats = num_components, axis=0)
            density_estimates[n]['mu_array'] = np.dot(D, density_estimates[n]['mu_array'] - means_stacked_d.T)
            for k in range(num_components): # loop over components to transform covariance matrices
                density_estimates[n]['Sigma_array'][k, :, :] = np.dot(D, np.dot(density_estimates[n]['Sigma_array'][k, :, :], D.T))

        
        return data_box_new, shifts, scalings

    def project_density_estimates(self, V):
        '''Return new databox that is identical to self but all density estimates have been projected onto the span of V'''

        # make copy of data box
        data_box_new = deepcopy(self)

        # extract from self
        num_examples = len(data_box_new.get_labels())
        density_estimates = data_box_new.get_density_estimates()

        # extract number of columns from V
        num_cols_V = np.shape(V)[1]

        # loop over examples
        density_estimates_new = {}
        for n in range(num_examples):
            mu_array_projected = np.dot(V.T, density_estimates[n]['mu_array'])
            num_comps = np.shape(density_estimates[n]['Sigma_array'])[0]
            Sigma_array_projected = np.zeros((num_comps, num_cols_V, num_cols_V))
            for k in range(num_comps):
                Sigma_array_projected[k, :, :] = np.dot(V.T, np.dot(density_estimates[n]['Sigma_array'][k, :, :], V))
            
            density_estimate = {}
            density_estimate['mu_array'] = mu_array_projected
            density_estimate['Sigma_array'] = Sigma_array_projected
            density_estimate['mix_weights'] = np.copy(density_estimates[n]['mix_weights'])

            density_estimates_new[n] = density_estimate

        # set density estimates of new data box
        data_box_new.set_density_estimates(density_estimates_new)

        return data_box_new

    def plot_posterior_samples(self, num_bins, smoothness, colours, labels=None, limits=None, title=None, legend=None, figsize=None):

        """ plots cornerplot for posterior samples, does not take into account weighting """
        
        # close all previous plots
        plt.close('all')

        # merge samples of each class
        class_samples = self._merge_samples_by_class()

        # extract number of dimensions
        num_dim = class_samples[0].shape[1]

        # loop over classes and plot corner-plots with different colours
        for idx in range(len(class_samples)):

            if idx == 0:
                figure = corner.corner(class_samples[idx], 
                                    bins=num_bins, 
                                    smooth=smoothness,
                                    #smooth1d = 0.5, 
                                    color=colours[idx],
                                    show_titles=True,
                                    labels=labels
                                    )
            else:
                figure = corner.corner(class_samples[idx], 
                                    bins=num_bins, 
                                    smooth=smoothness,
                                    #smooth1d = 0.5, 
                                    color=colours[idx],
                                    show_titles=True,
                                    labels=labels,
                                    fig=figure
                                    )
                
        # set plotting limits according to the specifications in limits
        if limits is not None:
            for j in range(num_dim):
                for i in range(num_dim):
                    
                    # set xlims
                    figure.axes[j * num_dim + i].set_xlim(limits[i, :])

                    # set ylims for subdiagonals
                    if i < j:
                        figure.axes[j * num_dim + i].set_ylim(limits[j, :])
        
        # set title
        if title is not None:
            figure.axes[round(num_dim / 2)].set_title(title)

        # set legend
        if legend is not None:
            figure.axes[0].legend(legend)

        if figsize is not None:
            plt.figure(figsize=figsize)

        return figure