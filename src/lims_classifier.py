import numpy as np
import logging
import scipy
from numba import njit, prange
from datetime import datetime
from src.utility import create_masked_arrays
from copy import copy

class LiMSClassifier:

    ### attributes ###
    classifier_opts = None

    ### methods ###
    def __init__(self, classifier_opts):
        self.classifier_opts = copy(classifier_opts) 

    def eval_pht_classifier(self, data_box_train, data_box_test, class_prior, V):
        
        # create masked arrays for training and test sets
        masked_arrays_train = create_masked_arrays(data_box_train)
        masked_arrays_test = create_masked_arrays(data_box_test)

        # extract masked arrays from the training set
        num_comps_vec_train = masked_arrays_train['num_comps_vec']

        mu_array_mask_train = masked_arrays_train['mu_array']
        Sigma_array_mask_train = masked_arrays_train['Sigma_array']
        mix_weights_mask_train = masked_arrays_train['mix_weights']

        num_pat_per_class_vec_train = masked_arrays_train['num_pat_per_class_vec']
        class_idx_mat_train = masked_arrays_train['class_idx_mat']

        # extract masked arrays from the test set
        num_samples_vec_test = masked_arrays_test['num_samples_vec']
        samples_mask_test = masked_arrays_test['samples']
        sample_weights_mask_test = masked_arrays_test['sample_weights']

        # check covariance structure
        if data_box_train.get_density_est_opts()['covariance_type'] == 'full' or data_box_train.get_density_est_opts()['covariance_type'] == 'diag':
            # call to NUMBA implemention
            predictions = self._bottleneck_full(V, class_prior, num_samples_vec_test, samples_mask_test, num_comps_vec_train, sample_weights_mask_test, mu_array_mask_train, Sigma_array_mask_train, mix_weights_mask_train, num_pat_per_class_vec_train, class_idx_mat_train)
        elif data_box_train.get_density_est_opts()['covariance_type'] == 'spherical':
            # call to NUMBA implemention
            predictions = self._bottleneck_spherical(V, class_prior, num_samples_vec_test, samples_mask_test, num_comps_vec_train, sample_weights_mask_test, mu_array_mask_train, Sigma_array_mask_train, mix_weights_mask_train, num_pat_per_class_vec_train, class_idx_mat_train)
        
        return predictions

    @staticmethod
    @njit(parallel=True)
    def _bottleneck_spherical(V, class_prior, num_samples_vec_test, samples_test, num_comps_vec_train, sample_weights_test, mu_array_train, Sigma_array_train, mix_weights_train, num_pat_per_class_vec_train, class_idx_mat_train):
        '''NUMBA implementation of computational bottleneck for spherical covariance matrices'''   

        # find number of patients and classes
        num_examples_test = np.shape(num_samples_vec_test)[0]
        num_classes = np.shape(class_prior)[0]
        num_col_V = np.shape(V)[1]

        # find determinant and inverse of V^T * V
        det_VtV = np.linalg.det(np.dot(np.transpose(V), V))
        inv_VtV = np.linalg.inv(np.dot(np.transpose(V), V))
        
        # initialize predictions matrix
        predictions = np.zeros((num_examples_test, num_classes))

        for n in prange(num_examples_test):
            
            prediction = np.zeros(num_classes)
            num_samples = num_samples_vec_test[n]

            for s in range(num_samples):
                
                c_sum_vec = np.zeros(num_classes)

                for c in range(num_classes):

                    m_k_sum_val = 0.

                    idx_label_c = class_idx_mat_train[0:num_pat_per_class_vec_train[c], c]
                    num_pat_in_class_c = num_pat_per_class_vec_train[c]

                    for mm in range(num_pat_in_class_c):
                        m = idx_label_c[mm]
                        num_mix_comp = num_comps_vec_train[m]
                        
                        for k in range(num_mix_comp):

                            theta = samples_test[n, :, s]
                            mu = mu_array_train[m, :, k]
                            sigma_sq = Sigma_array_train[m, k]
                            mix_weight = mix_weights_train[m, k]

                            a = theta - mu
                            Vta = np.dot(np.transpose(V), a)
                            det_VtSigmaV = (sigma_sq ** num_col_V) * det_VtV
                            inv_VtSigmaVVta = np.dot((1 / sigma_sq) * inv_VtV, Vta)
                            mvn_pdf_eval = (2 * np.pi) ** (-num_col_V / 2) * det_VtSigmaV ** (-0.5) * np.exp(-0.5 * np.dot(np.transpose(Vta), inv_VtSigmaVVta))
            
                            m_k_sum_val += mix_weight * mvn_pdf_eval

                    c_sum_vec[c] = m_k_sum_val * (class_prior[c] / num_pat_in_class_c)

                # extract sample weight and update class predictions
                alpha = sample_weights_test[n, s]
                prediction += alpha * (c_sum_vec / np.sum(c_sum_vec))

            predictions[n, :] = prediction
        
        return predictions
    
    @staticmethod
    @njit(parallel=True)
    def _bottleneck_full(V, class_prior, num_samples_vec_test, samples_test, num_comps_vec_train, sample_weights_test, mu_array_train, Sigma_array_train, mix_weights_train, num_pat_per_class_vec_train, class_idx_mat_train):
        '''NUMBA implementation of computational bottleneck for full covariance matrices'''                              
        
        # find number of patients and classes
        num_examples_test = np.shape(num_samples_vec_test)[0]
        num_classes = np.shape(class_prior)[0]
        num_col_V = np.shape(V)[1]
        
        # initialize predictions matrix
        predictions = np.zeros((num_examples_test, num_classes))

        for n in prange(num_examples_test):
            
            prediction = np.zeros(num_classes)
            num_samples = num_samples_vec_test[n]

            for s in range(num_samples):
                
                c_sum_vec = np.zeros(num_classes)

                for c in range(num_classes):

                    m_k_sum_val = 0.
                    idx_label_c = class_idx_mat_train[0:num_pat_per_class_vec_train[c], c]
                    num_pat_in_class_c = num_pat_per_class_vec_train[c]

                    for mm in range(num_pat_in_class_c):
                        m = idx_label_c[mm]
                        num_mix_comp = num_comps_vec_train[m]
                        
                        for k in range(num_mix_comp):

                            theta = samples_test[n, :, s]
                            mu = mu_array_train[m, :, k]
                            Sigma = np.copy(Sigma_array_train[m, k, :, :])                    # NOTE this could potentially be improved
                            mix_weight = mix_weights_train[m, k]

                            a = theta - mu
                            Vta = np.dot(np.transpose(V), a)
                            VtSigmaV = np.dot(np.transpose(V), np.dot(Sigma, V))

                            det_VtSigmaV = np.linalg.det(VtSigmaV)
                            inv_VtSigmaVVta = np.linalg.solve(VtSigmaV, Vta)

                            mvn_pdf_eval = (2 * np.pi) ** (-num_col_V / 2) * det_VtSigmaV ** (-0.5) * np.exp(-0.5 * np.dot(np.transpose(Vta), inv_VtSigmaVVta))
            
                            m_k_sum_val += mix_weight * mvn_pdf_eval

                    c_sum_vec[c] = m_k_sum_val * (class_prior[c] / num_pat_in_class_c)

                # extract sample weight and update class predictions
                alpha = sample_weights_test[n, s]
                prediction += alpha * (c_sum_vec / np.sum(c_sum_vec))

            predictions[n, :] = prediction
        
        return predictions