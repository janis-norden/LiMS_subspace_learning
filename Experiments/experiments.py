# import external modules
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import time
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklvq import GMLVQ
import matplotlib.pyplot as plt

# import own modules
from src.data_box import DataBox
from src.subspace_learner import SubspaceLearner
from src.lims_classifier import LiMSClassifier
from src.density_estimator import DensityEstimator
from src.utility import set_class_prior
import src.config as config

'''

    Experiment 1  : prednisone spiral, compare different covariance types for subspace likelihood cost function
    Experiment 2  : prednisone spiral, compare batch vs. iterative optimization
    Experiment 3  : prednisone spiral, spherical subspace likelihood cost vs. GLVQ cost
    Experiment 4a : gravitational waves, k-fold cross validation, spherical covariances, learn with subspace likelihood cost and GLVQ cost
    Experiment 4b : gravitational waves, k-fold random subsampling, spherical covariances, learn with subspace likelihood cost and GLVQ cost
    Experiment 5  : prednisone spiral, spherical covariances, learn subspaces from worst case initiaizations
    Experiment 6  : prednisone spiral, spherical covariances, compare how much PHT adds on top of GMLVQ
    
'''


def run_exp1_pred_covariances_types(exp1_opts):
    '''

    Description
    ----------
    In this experiment, the inflcuence of the underlying density estimate on the quality of the subspace learning is investigated.
    Based on the two density estimates (spherical covariances vs. full covariances), three subspace learners are tested against each other.
    The subspace learners are subspace likelihood (ssl) spherical, ssl full and ssl full approximated.
    

    
    Parameters
    ----------
    exp1_opts : dict
        Dict containing the following parameters:

    num_rand_init : int
        positive int, number of random initializations per subspace learning

    cols_vec : array
        np array int, indicates the dimensions of subspaces to search

    file_name_spherical : str
        name of file in data/interim/ to be loaded as density est. with spherical cov. matrices

    file_name_full : str
        name of file in data/interim/ to be loaded as density est. with full cov. matrices


    '''

    # start timer
    t0 = time.time()

    # extract info from exp1_opts
    num_rand_init = exp1_opts['num_rand_init']
    cols_vec = exp1_opts['cols_vec']

    # load data from previous density estimation
    file_name_spherical = exp1_opts['file_name_spherical']
    data_box_spherical = DataBox()
    data_box_spherical = data_box_spherical.load_data('/interim/' + file_name_spherical)

    file_name_full = exp1_opts['file_name_full']
    data_box_full = DataBox()
    data_box_full = data_box_full.load_data('/interim/' + file_name_full)

    # extract number of parameters
    num_dim_ambient = data_box_spherical._dynamical_system.num_params
    ds_name = data_box_spherical._dynamical_system.name
    
    # make 50/50 train and test split
    labels = data_box_spherical.get_labels()
    skf = StratifiedKFold(n_splits = 2, shuffle=True, random_state=0)
    split1, split2 = skf.split(np.zeros(len(labels)), labels)
    train_index = split1[0]
    test_index = split1[1]
    data_box_spherical_train = data_box_spherical.select_examples(train_index)
    data_box_spherical_test = data_box_spherical.select_examples(test_index)
    data_box_spherical_train_norm, means, std_devs = data_box_spherical_train.shift_and_scale_data()       # apply standard-score transform
    data_box_spherical_test_norm, _, _ = data_box_spherical_test.shift_and_scale_data(means, std_devs)     # apply the same transform to test data

    data_box_full_train = data_box_full.select_examples(train_index)
    data_box_full_test = data_box_full.select_examples(test_index) 
    data_box_full_train_norm, means, std_devs = data_box_full_train.shift_and_scale_data()       # apply standard-score transform
    data_box_full_test_norm, _, _ = data_box_full_test.shift_and_scale_data(means, std_devs)     # apply the same transform to test data

    # set general subspace learning options
    subspace_learn_opts = {
        'subspace_dim':                                  None,      # positive integer      
        'cost_function':                'subspace_likelihood',      # 'subspace_likelihood', 'GLVQ', 'subspace_likelihood_gradient'
        'ssl_approx':                                    None,      # boolean, only makes a different if 'subspace_likelihood' is selected
        'opt_mode':                                   'batch',      # 'batch', 'iterative'
        'init_type':                                 'random',      # 'PCA', 'random'
        'class_prior':                                 'flat',      # 'empirical', 'flat'
        'scipy_min_method':                            'BFGS',      # 'BFGS'
        'scipy_min_tol':                                1e-05,      # positive float
        'scipy_min_maxiter':                              100,      # positive int
        'scipy_min_disp':                                True,      # boolean
    }

    # define classifier
    classifier = LiMSClassifier(None)

    # set class prior
    class_prior = set_class_prior(data_box_spherical.get_labels(), subspace_learn_opts['class_prior'])

    # loop over the desired number of columns
    outcomes = {}
    for i in range(len(cols_vec)):

        # set the desired number of columns
        subspace_learn_opts['subspace_dim'] = cols_vec[i]

        # loop over random initializations
        subspace_learn_opts['init_type'] = 'random'
        current_dim_results = {'spherical': [], 'full_approx': [], 'full': []}
        for trial in range(num_rand_init):

            ssl = SubspaceLearner(data_box_spherical_train_norm, subspace_learn_opts)                                                          # spherical
            t0_ssl = time.time()
            db_spherical = ssl.run_subspace_learning()
            tend_ssl = time.time()
            subspace_learned = db_spherical.get_subspace_learned()
            pht_predictions_train = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_train_norm, class_prior, subspace_learned['V_opt'])
            pht_predictions_test = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_test_norm, class_prior, subspace_learned['V_opt'])

            current_dim_results['spherical'].append({'subspace_learned': subspace_learned, 
                                                     'pht_predictions_train': pht_predictions_train, 
                                                     'pht_predictions_test': pht_predictions_test, 
                                                     'ssl_time': tend_ssl - t0_ssl})
            
            subspace_learn_opts['ssl_approx'] = True                                                                                      # full approx
            ssl = SubspaceLearner(data_box_full_train_norm, subspace_learn_opts)                  
            t0_ssl = time.time()
            db_full_approx = ssl.run_subspace_learning()
            tend_ssl = time.time()
            subspace_learned = db_full_approx.get_subspace_learned()
            pht_predictions_train = classifier.eval_pht_classifier(data_box_full_train_norm, data_box_full_train_norm, class_prior, subspace_learned['V_opt'])
            pht_predictions_test = classifier.eval_pht_classifier(data_box_full_train_norm, data_box_full_test_norm, class_prior, subspace_learned['V_opt'])
            
            current_dim_results['full_approx'].append({'subspace_learned': subspace_learned, 
                                                     'pht_predictions_train': pht_predictions_train, 
                                                     'pht_predictions_test': pht_predictions_test, 
                                                     'ssl_time': tend_ssl - t0_ssl})

            subspace_learn_opts['ssl_approx'] = False                                                                               # full
            ssl = SubspaceLearner(data_box_full_train_norm, subspace_learn_opts)                  
            t0_ssl = time.time()
            db_full_approx = ssl.run_subspace_learning()
            tend_ssl = time.time()
            subspace_learned = db_full_approx.get_subspace_learned()
            pht_predictions_train = classifier.eval_pht_classifier(data_box_full_train_norm, data_box_full_train_norm, class_prior, subspace_learned['V_opt'])
            pht_predictions_test = classifier.eval_pht_classifier(data_box_full_train_norm, data_box_full_test_norm, class_prior, subspace_learned['V_opt'])
            
            current_dim_results['full'].append({'subspace_learned': subspace_learned, 
                                                'pht_predictions_train': pht_predictions_train, 
                                                'pht_predictions_test': pht_predictions_test, 
                                                'ssl_time': tend_ssl - t0_ssl})
            
        # run ssl with GMLVQ initalization
        subspace_learn_opts['init_type'] = 'GMLVQ'

        ssl = SubspaceLearner(data_box_spherical_train_norm, subspace_learn_opts)                                                          # spherical
        t0_ssl = time.time()
        db_spherical = ssl.run_subspace_learning()
        tend_ssl = time.time()
        subspace_learned = db_spherical.get_subspace_learned()
        pht_predictions_train = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_train_norm, class_prior, subspace_learned['V_opt'])
        pht_predictions_test = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_test_norm, class_prior, subspace_learned['V_opt'])

        current_dim_results['spherical'].append({'subspace_learned': subspace_learned, 
                                                    'pht_predictions_train': pht_predictions_train, 
                                                    'pht_predictions_test': pht_predictions_test, 
                                                    'ssl_time': tend_ssl - t0_ssl})
        
        subspace_learn_opts['ssl_approx'] = True                                                                                # full approx
        ssl = SubspaceLearner(data_box_full_train_norm, subspace_learn_opts)                  
        t0_ssl = time.time()
        db_full_approx = ssl.run_subspace_learning()
        tend_ssl = time.time()
        subspace_learned = db_full_approx.get_subspace_learned()
        pht_predictions_train = classifier.eval_pht_classifier(data_box_full_train_norm, data_box_full_train_norm, class_prior, subspace_learned['V_opt'])
        pht_predictions_test = classifier.eval_pht_classifier(data_box_full_train_norm, data_box_full_test_norm, class_prior, subspace_learned['V_opt'])
        
        current_dim_results['full_approx'].append({'subspace_learned': subspace_learned, 
                                                    'pht_predictions_train': pht_predictions_train, 
                                                    'pht_predictions_test': pht_predictions_test, 
                                                    'ssl_time': tend_ssl - t0_ssl})

        subspace_learn_opts['ssl_approx'] = False                                                                               # full
        ssl = SubspaceLearner(data_box_full_train_norm, subspace_learn_opts)                  
        t0_ssl = time.time()
        db_full_approx = ssl.run_subspace_learning()
        tend_ssl = time.time()
        subspace_learned = db_full_approx.get_subspace_learned()
        pht_predictions_train = classifier.eval_pht_classifier(data_box_full_train_norm, data_box_full_train_norm, class_prior, subspace_learned['V_opt'])
        pht_predictions_test = classifier.eval_pht_classifier(data_box_full_train_norm, data_box_full_test_norm, class_prior, subspace_learned['V_opt'])
        
        current_dim_results['full'].append({'subspace_learned': subspace_learned, 
                                            'pht_predictions_train': pht_predictions_train, 
                                            'pht_predictions_test': pht_predictions_test, 
                                            'ssl_time': tend_ssl - t0_ssl})

        outcomes[i] = current_dim_results

    # record results for ambient dimension
    current_dim_results = {'spherical': [], 'full_approx': [], 'full': []}

    pht_predictions_train = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_train_norm, class_prior, np.identity(num_dim_ambient))       # spherical
    pht_predictions_test = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_test_norm, class_prior, np.identity(num_dim_ambient))
    
    current_dim_results['spherical'].append({'subspace_learned': np.identity(num_dim_ambient), 
                                            'pht_predictions_train': pht_predictions_train, 
                                            'pht_predictions_test': pht_predictions_test, 
                                            'ssl_time': 0})
    
    pht_predictions_train = classifier.eval_pht_classifier(data_box_full_train_norm, data_box_full_train_norm, class_prior, np.identity(num_dim_ambient))                 # full approx. and full
    pht_predictions_test = classifier.eval_pht_classifier(data_box_full_train_norm, data_box_full_test_norm, class_prior, np.identity(num_dim_ambient))

    for cost_func in ['full_approx', 'full']:
        current_dim_results[cost_func].append({'subspace_learned': np.identity(num_dim_ambient), 
                                            'pht_predictions_train': pht_predictions_train, 
                                            'pht_predictions_test': pht_predictions_test, 
                                            'ssl_time': 0})

    outcomes[len(cols_vec)] = current_dim_results

    # stop global timer
    tend = time.time()
    total_time = tend - t0

    # save outcomes to file
    path_to_project_root = str(Path(__file__).parents[1])                                   # find path to project root directory
    current_date_time = datetime.now()                                                      # find current time and date
    date_time_str = current_date_time.strftime("%Y_%m_%d_%H_%M_%S")
    save_path = path_to_project_root + '/data/processed/exp1_' + date_time_str + '_' + ds_name + '.pckl'    # determine path to Python/data/

    # create dict to collect save data
    save_data = {   
        'data_box_spherical': data_box_spherical,
        'data_box_full': data_box_full,
        'train_index': train_index,
        'test_index': test_index,
        'exp1_opts': exp1_opts, 
        'subspace_learn_opts': subspace_learn_opts,
        'outcomes': outcomes, 
        'total_time': total_time
        }

    # open a file, save data and close
    f = open(save_path, 'wb')
    pickle.dump(save_data, f)
    f.close()

def run_exp2_pred_batch_iterative(exp2_opts):
    
    '''
    Description
    ----------
    In this experiment, the effect of the optimization type (batch or iterative) on the quality of the subspace learning is investigated.
    Based on the spherical density estimate two subspace learners are tested against each other.
    The subspace learners are subspace likelihood (ssl) spherical, once with batch optimization and once with iterative optimization
    The function automatically saves results to the 'data/processed/' folder and files names start with 'exp2_'.
    
    Parameters
    ----------
    exp2_opts : dict
        Dict containing the following parameters:

    num_rand_init : int
        positive int, number of random initializations per subspace learning

    cols_vec : array
        np array int, indicates the dimensions of subspaces to search

    file_name_spherical : str
        name of file in data/interim/ to be loaded as density est. with spherical cov. matrices

    '''

    # start timer
    t0 = time.time()

    # extract info from exp_opts
    num_rand_init = exp2_opts['num_rand_init']
    cols_vec = exp2_opts['cols_vec']

    # load data from previous density estimation
    file_name_spherical = exp2_opts['file_name_spherical']
    data_box_spherical = DataBox()
    data_box_spherical = data_box_spherical.load_data('/interim/' + file_name_spherical)

    # extract number of parameters
    num_dim_ambient = data_box_spherical._dynamical_system.num_params
    ds_name = data_box_spherical._dynamical_system.name

    # make 50/50 train and test split
    labels = data_box_spherical.get_labels()
    skf = StratifiedKFold(n_splits = 2, shuffle=True, random_state=0)
    split1, split2 = skf.split(np.zeros(len(labels)), labels)
    train_index = split1[0]
    test_index = split1[1]
    data_box_spherical_train = data_box_spherical.select_examples(train_index)
    data_box_spherical_test = data_box_spherical.select_examples(test_index)
    data_box_spherical_train_norm, means, std_devs = data_box_spherical_train.shift_and_scale_data()       # apply standard-score transform
    data_box_spherical_test_norm, _, _ = data_box_spherical_test.shift_and_scale_data(means, std_devs)     # apply the same transform to test data

    # set general subspace learning options
    subspace_learn_opts = {
        'subspace_dim':                                  None,      # positive integer      
        'cost_function':                'subspace_likelihood',      # 'subspace_likelihood', 'GLVQ', 'subspace_likelihood_gradient'
        'ssl_approx':                                    None,      # boolean, only makes a different if 'subspace_likelihood' is selected and the density estimates are full-cov.
        'opt_mode':                                      None,      # 'batch', 'iterative'
        'init_type':                                     None,      # 'PCA', 'random'
        'class_prior':                                 'flat',      # 'empirical', 'flat'
        'scipy_min_method':                            'BFGS',      # 'BFGS'
        'scipy_min_tol':                                1e-05,      # positive float
        'scipy_min_maxiter':                              100,      # positive int
        'scipy_min_disp':                                True,      # boolean
    }

    # define classifier
    classifier = LiMSClassifier(None)

    # find class prior
    class_prior = set_class_prior(data_box_spherical.get_labels(), subspace_learn_opts['class_prior'])

    # loop over the desired number of columns
    outcomes = {}
    for i in range(len(cols_vec)):

        # set the desired number of columns
        subspace_learn_opts['subspace_dim'] = cols_vec[i]

        # loop over random initializations                                                 # random initializatoins
        subspace_learn_opts['init_type'] = 'random'
        current_dim_results = {'batch': [], 'iterative': []}
        for trial in range(num_rand_init):

            # loop over optimization modi
            for opt_mode in ['batch', 'iterative']:
                
                # set optimization mode
                subspace_learn_opts['opt_mode'] = opt_mode
                
                # run subspace learning
                ssl = SubspaceLearner(data_box_spherical_train_norm, subspace_learn_opts)
                t0_ssl = time.time()
                db_spherical = ssl.run_subspace_learning()
                tend_ssl = time.time()
                subspace_learned = db_spherical.get_subspace_learned()
                pht_predictions_train = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_train_norm, class_prior, subspace_learned['V_opt'])
                pht_predictions_test = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_test_norm, class_prior, subspace_learned['V_opt'])

                # record results
                current_dim_results[opt_mode].append({'subspace_learned': subspace_learned, 
                                                        'pht_predictions_train': pht_predictions_train, 
                                                        'pht_predictions_test': pht_predictions_test, 
                                                        'ssl_time': tend_ssl - t0_ssl})


        # run ssl with GMLVQ initalization
        subspace_learn_opts['init_type'] = 'GMLVQ'                                           # GMLVQ initialization

        # loop over optimization modi
        for opt_mode in ['batch', 'iterative']:
            
            # set optimization mode
            subspace_learn_opts['opt_mode'] = opt_mode

            # run subspace learning
            ssl = SubspaceLearner(data_box_spherical_train_norm, subspace_learn_opts)
            t0_ssl = time.time()
            db_spherical = ssl.run_subspace_learning()
            tend_ssl = time.time()
            subspace_learned = db_spherical.get_subspace_learned()
            pht_predictions_train = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_train_norm, class_prior, subspace_learned['V_opt'])
            pht_predictions_test = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_test_norm, class_prior, subspace_learned['V_opt'])

            # record results
            current_dim_results[opt_mode].append({'subspace_learned': subspace_learned, 
                                                    'pht_predictions_train': pht_predictions_train, 
                                                    'pht_predictions_test': pht_predictions_test, 
                                                    'ssl_time': tend_ssl - t0_ssl})

        outcomes[i] = current_dim_results

    # record results for ambient dimension
    pht_predictions_train = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_train_norm, class_prior, np.identity(num_dim_ambient))
    pht_predictions_test = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_test_norm, class_prior, np.identity(num_dim_ambient))
    
    current_dim_results = {'batch': [], 'iterative': []}
    for opt_mode in ['batch', 'iterative']:
        current_dim_results[opt_mode].append({'subspace_learned': np.identity(num_dim_ambient), 
                                            'pht_predictions_train': pht_predictions_train, 
                                            'pht_predictions_test': pht_predictions_test, 
                                            'ssl_time': 0})

    outcomes[len(cols_vec)] = current_dim_results

    # stop global timer
    tend = time.time()
    total_time = tend - t0

    # save outcomes to file
    path_to_project_root = str(Path(__file__).parents[1])                                   # find path to project root directory
    current_date_time = datetime.now()                                                      # find current time and date
    date_time_str = current_date_time.strftime("%Y_%m_%d_%H_%M_%S")
    save_path = path_to_project_root + '/data/processed/exp2_' + date_time_str + '_' + ds_name + '.pckl'    # determine path to Python/data/

    # create dict to collect save data
    save_data = {   
        'data_box_spherical': data_box_spherical,
        'train_index': train_index,
        'test_index': test_index,
        'exp2_opts': exp2_opts, 
        'subspace_learn_opts': subspace_learn_opts,
        'outcomes': outcomes, 
        'total_time': total_time
        }

    # open a file, save data and close
    f = open(save_path, 'wb')
    pickle.dump(save_data, f)
    f.close()

def run_exp3_pred_ssl_GLVQ(exp3_opts):

    '''
    Description
    ----------
    In this experiment, the effect of the choice of cost function type (subspace likelihood vs. heuristic GLVQ) on the quality of the subspace learning is investigated.
    Based on the spherical density estimate two subspace learners are tested against each other.
    The subspace learners are subspace likelihood (ssl) spherical and GLVQ spherical, both optimized with an iterative scheme.
    The function automatically saves results to the 'data/processed/' folder and files names start with 'exp3_'.
    
    Parameters
    ----------
    exp3_opts : dict
        Dict containing the following parameters:

    num_rand_init : int
        positive int, number of random initializations per subspace learning

    cols_vec : array
        np array int, indicates the dimensions of subspaces to search

    file_name_spherical : str
        name of file in data/interim/ to be loaded as density est. with spherical cov. matrices

    '''

    # start timer
    t0 = time.time()

    # extract info from exp_opts
    num_rand_init = exp3_opts['num_rand_init']
    cols_vec = exp3_opts['cols_vec']

    # load data from previous density estimation
    file_name_spherical = exp3_opts['file_name_spherical']
    data_box_spherical = DataBox()
    data_box_spherical = data_box_spherical.load_data('/interim/' + file_name_spherical)

    # extract number of parameters
    num_dim_ambient = data_box_spherical._dynamical_system.num_params
    ds_name = data_box_spherical._dynamical_system.name

    # make 50/50 train and test split
    labels = data_box_spherical.get_labels()
    skf = StratifiedKFold(n_splits = 2, shuffle=True, random_state=42)
    split1, split2 = skf.split(np.zeros(len(labels)), labels)
    train_index = split1[0]
    test_index = split1[1]
    data_box_spherical_train = data_box_spherical.select_examples(train_index)
    data_box_spherical_test = data_box_spherical.select_examples(test_index)
    data_box_spherical_train_norm, means, std_devs = data_box_spherical_train.shift_and_scale_data()       # apply standard-score transform
    data_box_spherical_test_norm, _, _ = data_box_spherical_test.shift_and_scale_data(means, std_devs)     # apply the same transform to test data

    # set general subspace learning options
    subspace_learn_opts = {
        'subspace_dim':                                  None,      # positive integer      
        'cost_function':                                 None,      # 'subspace_likelihood', 'GLVQ', 'subspace_likelihood_gradient'
        'ssl_approx':                                    None,      # boolean, only makes a different if 'subspace_likelihood' is selected and the density estimates are full-cov.
        'opt_mode':                               'iterative',      # 'batch', 'iterative'
        'init_type':                                     None,      # 'PCA', 'random'
        'class_prior':                                 'flat',      # 'empirical', 'flat'
        'scipy_min_method':                            'BFGS',      # 'BFGS'
        'scipy_min_tol':                                1e-05,      # positive float
        'scipy_min_maxiter':                              100,      # positive int
        'scipy_min_disp':                                True,      # boolean
    }

    # define classifier
    classifier = LiMSClassifier(None)

    # find class prior
    class_prior = set_class_prior(data_box_spherical.get_labels(), subspace_learn_opts['class_prior'])

    # loop over the desired number of columns
    outcomes = {}
    for i in range(len(cols_vec)):

        # set the desired number of columns
        subspace_learn_opts['subspace_dim'] = cols_vec[i]

        # loop over random initializations                                                 # random initializatoins
        subspace_learn_opts['init_type'] = 'random'
        current_dim_results = {'subspace_likelihood': [], 'GLVQ': []}
        for trial in range(num_rand_init):

            # loop over different cost functions
            for cost_func_type in ['subspace_likelihood', 'GLVQ']:
                
                # set cost function for subspace learning
                subspace_learn_opts['cost_function'] = cost_func_type
                
                # run subspace learning
                ssl = SubspaceLearner(data_box_spherical_train_norm, subspace_learn_opts)
                t0_ssl = time.time()
                db_spherical = ssl.run_subspace_learning()
                tend_ssl = time.time()
                subspace_learned = db_spherical.get_subspace_learned()
                pht_predictions_train = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_train_norm, class_prior, subspace_learned['V_opt'])
                pht_predictions_test = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_test_norm, class_prior, subspace_learned['V_opt'])

                # record results
                current_dim_results[cost_func_type].append({'subspace_learned': subspace_learned, 
                                                        'pht_predictions_train': pht_predictions_train, 
                                                        'pht_predictions_test': pht_predictions_test, 
                                                        'ssl_time': tend_ssl - t0_ssl})


        # run ssl with GMLVQ initalization
        subspace_learn_opts['init_type'] = 'GMLVQ'                                           # GMLVQ initialization

        # loop over different cost functions
        for cost_func_type in ['subspace_likelihood', 'GLVQ']:
        
            # run subspace learning
                ssl = SubspaceLearner(data_box_spherical_train_norm, subspace_learn_opts)
                t0_ssl = time.time()
                db_spherical = ssl.run_subspace_learning()
                tend_ssl = time.time()
                subspace_learned = db_spherical.get_subspace_learned()
                pht_predictions_train = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_train_norm, class_prior, subspace_learned['V_opt'])
                pht_predictions_test = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_test_norm, class_prior, subspace_learned['V_opt'])

                # record results
                current_dim_results[cost_func_type].append({'subspace_learned': subspace_learned, 
                                                        'pht_predictions_train': pht_predictions_train, 
                                                        'pht_predictions_test': pht_predictions_test, 
                                                        'ssl_time': tend_ssl - t0_ssl})

        outcomes[i] = current_dim_results
    
    # record results for ambient dimension
    pht_predictions_train = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_train_norm, class_prior, np.identity(num_dim_ambient))
    pht_predictions_test = classifier.eval_pht_classifier(data_box_spherical_train_norm, data_box_spherical_test_norm, class_prior, np.identity(num_dim_ambient))
    
    current_dim_results = {'GLVQ': [], 'subspace_likelihood': []}
    for cost_func_type in ['subspace_likelihood', 'GLVQ']:
        current_dim_results[cost_func_type].append({'subspace_learned': np.identity(num_dim_ambient), 
                                            'pht_predictions_train': pht_predictions_train, 
                                            'pht_predictions_test': pht_predictions_test, 
                                            'ssl_time': 0})

    outcomes[len(cols_vec)] = current_dim_results

    # stop timer
    tend = time.time()
    total_time = tend - t0

    # save outcomes to file
    path_to_project_root = str(Path(__file__).parents[1])                                   # find path to project root directory
    current_date_time = datetime.now()                                                      # find current time and date
    date_time_str = current_date_time.strftime("%Y_%m_%d_%H_%M_%S")
    save_path = path_to_project_root + '/data/processed/exp3_' + date_time_str + '_' + ds_name + '.pckl'    # determine path to Python/data/

    # create dict to collect save data
    save_data = {   
        'data_box_spherical': data_box_spherical,
        'train_index': train_index,
        'test_index': test_index,
        'exp3_opts': exp3_opts, 
        'subspace_learn_opts': subspace_learn_opts,
        'outcomes': outcomes, 
        'total_time': total_time
        }

    # open a file, save data and close
    f = open(save_path, 'wb')
    pickle.dump(save_data, f)
    f.close()

def run_exp4a_gw_spherical(exp4a_opts):

    '''
    Description
    ----------
    In this experiment, the PHT classifier and subspace learning are applied to a 3-class classification problem of Graviatational Waves.
    Spherical Gaussian Mixture models have been fitted to the posterior distributions.
    The subspace learners in use are subspace likelihood (ssl) and GLVQ heuristic cost, both optimized with an iterative scheme. 
    To estimate the out-of-sample performance, k-fold cross-validation is used.
    The function automatically saves results to the 'data/processed/' folder and files names start with 'exp4a_'.
    
    Parameters
    ----------
    exp4a_opts : dict
        Dict containing the following parameters:

    num_folds : dict
        positive int, number of folds for k-fold cross-validation

    num_rand_init : int
        positive int, number of random initializations per subspace learning

    cols_vec : array
        np array int, indicates the dimensions of subspaces to search

    file_name_spherical : str
        name of file in data/interim/ to be loaded as density est. with spherical cov. matrices

    '''

    # start timer
    t0 = time.time()
    data_time_at_start = datetime.now()                                                      

    # extract info from exp_opts
    num_rand_init = exp4a_opts['num_rand_init']
    cols_vec = exp4a_opts['cols_vec']
    num_folds = exp4a_opts['num_folds']

    # load data from previous density estimation
    file_name_spherical = exp4a_opts['file_name_spherical']
    data_box_spherical = DataBox()
    data_box_spherical = data_box_spherical.load_data('/interim/' + file_name_spherical)

    # find path to project root directory
    path_to_project_root = str(Path(__file__).parents[1])                                   

    # extract info from data box
    labels = data_box_spherical.get_labels()
    num_dim_ambient = data_box_spherical._dynamical_system.num_params
    num_examples = len(labels)

    # set general subspace learning options
    subspace_learn_opts = {
        'subspace_dim':                                  None,      # positive integer      
        'cost_function':                                 None,      # 'subspace_likelihood', 'GLVQ', 'subspace_likelihood_gradient'
        'ssl_approx':                                    None,      # boolean, only makes a different if 'subspace_likelihood' is selected and the density estimates are full-cov.
        'opt_mode':                               'iterative',      # 'batch', 'iterative'
        'init_type':                                     None,      # 'PCA', 'random'
        'class_prior':                                 'flat',      # 'empirical', 'flat'
        'scipy_min_method':                            'BFGS',      # 'BFGS'
        'scipy_min_tol':                                1e-04,      # positive float
        'scipy_min_maxiter':                              100,      # positive int
        'scipy_min_disp':                                True,      # boolean
    }

    # set density estimation options
    density_est_opts = {
        'type':             'bayesian_gaussian_mixture',
        'covariance_type':                  'spherical',    # 'full', 'diag' 'spherical' 
        'n_components':                              10,    # 10
        'n_init' :                                    5,    # 15
        'max_iter':                                 500,    # 200
        'trim_percent':                            0.99
    }

    # find class prior
    class_prior = set_class_prior(data_box_spherical.get_labels(), subspace_learn_opts['class_prior'])

    # define classifier
    classifier = LiMSClassifier(None)
    
    # set up k-fold cross-validation
    skf = StratifiedKFold(n_splits=num_folds)

    # create global variable to be passed to callback function
    global OPT_INFO
    OPT_INFO = config.OPT_INFO

    # loop over folds
    outcomes_folds = {}
    for fold_idx, (train_index, test_index) in enumerate(skf.split(np.zeros(num_examples), labels)):

        # update fold to be printed to screen
        OPT_INFO['fold'] = fold_idx + 1

        # initialize dict to hold results for current fold
        dimension_results = {}
        
        # create training and testing data sets
        data_box_train = data_box_spherical.select_examples(train_index)
        data_box_test = data_box_spherical.select_examples(test_index)

        # normalize training and test data
        data_box_train_norm, means, std_devs = data_box_train.shift_and_scale_data()       # apply standard-score transform
        data_box_test_norm, _, _ = data_box_test.shift_and_scale_data(means, std_devs)     # apply the same transform to test data

        # define density estimator object
        dens_est_train = DensityEstimator(data_box_train_norm, density_est_opts)
        dens_est_test = DensityEstimator(data_box_test_norm, density_est_opts)

        # run density estimation
        data_box_train_norm = dens_est_train.run_density_estimation()
        data_box_test_norm = dens_est_test.run_density_estimation()

        # loop over subspace dimensions
        for i in range(len(cols_vec)):

            # initialize dict to hold results for current dim
            current_dim_results = {'GLVQ': [], 'subspace_likelihood': []}

            # set the desired number of columns
            subspace_learn_opts['subspace_dim'] = cols_vec[i]

            # loop over random initializations                                                 # random initializations
            subspace_learn_opts['init_type'] = 'random'
            for trial in range(num_rand_init):
                
                # set cost function type to GLVQ
                subspace_learn_opts['cost_function'] = 'GLVQ'

                # run subspace learning
                ssl = SubspaceLearner(data_box_train_norm, subspace_learn_opts)
                t0_ssl = time.time()
                db_GLVQ = ssl.run_subspace_learning()
                tend_ssl = time.time()
                subspace_learned = db_GLVQ.get_subspace_learned()
                pht_predictions_train = classifier.eval_pht_classifier(data_box_train_norm, data_box_train_norm, class_prior, subspace_learned['V_opt'])
                pht_predictions_test = classifier.eval_pht_classifier(data_box_train_norm, data_box_test_norm, class_prior, subspace_learned['V_opt'])

                # record results
                current_dim_results['GLVQ'].append({'subspace_learned': subspace_learned, 
                                                        'pht_predictions_train': pht_predictions_train, 
                                                        'pht_predictions_test': pht_predictions_test, 
                                                        'ssl_time': tend_ssl - t0_ssl})


                # Subspace likelihood with warm start
                subspace_learn_opts['init_type'] = subspace_learned['V_opt']            # warm start                                                                                        
                subspace_learn_opts['cost_function'] = 'subspace_likelihood'

                # run subspace learning
                ssl = SubspaceLearner(data_box_train_norm, subspace_learn_opts)
                t0_ssl = time.time()
                db_subspace_likelihood = ssl.run_subspace_learning()
                tend_ssl = time.time()
                subspace_learned = db_subspace_likelihood.get_subspace_learned()
                pht_predictions_train = classifier.eval_pht_classifier(data_box_train_norm, data_box_train_norm, class_prior, subspace_learned['V_opt'])
                pht_predictions_test = classifier.eval_pht_classifier(data_box_train_norm, data_box_test_norm, class_prior, subspace_learned['V_opt'])

                # record results
                current_dim_results['subspace_likelihood'].append({'subspace_learned': subspace_learned, 
                                                        'pht_predictions_train': pht_predictions_train, 
                                                        'pht_predictions_test': pht_predictions_test, 
                                                        'ssl_time': tend_ssl - t0_ssl})

            # GMLVQ init
            subspace_learn_opts['init_type'] = 'GMLVQ'
            
            # set cost function type to GLVQ
            subspace_learn_opts['cost_function'] = 'GLVQ'

            # run subspace learning
            ssl = SubspaceLearner(data_box_train_norm, subspace_learn_opts)
            t0_ssl = time.time()
            db_GLVQ = ssl.run_subspace_learning()
            tend_ssl = time.time()
            subspace_learned = db_GLVQ.get_subspace_learned()
            pht_predictions_train = classifier.eval_pht_classifier(data_box_train_norm, data_box_train_norm, class_prior, subspace_learned['V_opt'])
            pht_predictions_test = classifier.eval_pht_classifier(data_box_train_norm, data_box_test_norm, class_prior, subspace_learned['V_opt'])

            # record results
            current_dim_results['GLVQ'].append({'subspace_learned': subspace_learned, 
                                                    'pht_predictions_train': pht_predictions_train, 
                                                    'pht_predictions_test': pht_predictions_test, 
                                                    'ssl_time': tend_ssl - t0_ssl})


            # Subspace likelihood with warm start
            subspace_learn_opts['init_type'] = subspace_learned['V_opt']            # warm start                                                                                        
            subspace_learn_opts['cost_function'] = 'subspace_likelihood'

            # run subspace learning
            ssl = SubspaceLearner(data_box_train_norm, subspace_learn_opts)
            t0_ssl = time.time()
            db_subspace_likelihood = ssl.run_subspace_learning()
            tend_ssl = time.time()
            subspace_learned = db_subspace_likelihood.get_subspace_learned()
            pht_predictions_train = classifier.eval_pht_classifier(data_box_train_norm, data_box_train_norm, class_prior, subspace_learned['V_opt'])
            pht_predictions_test = classifier.eval_pht_classifier(data_box_train_norm, data_box_test_norm, class_prior, subspace_learned['V_opt'])

            # record results
            current_dim_results['subspace_likelihood'].append({'subspace_learned': subspace_learned, 
                                                    'pht_predictions_train': pht_predictions_train, 
                                                    'pht_predictions_test': pht_predictions_test, 
                                                    'ssl_time': tend_ssl - t0_ssl})

            # record results of current dimension
            dimension_results[i] = current_dim_results

        # record results for ambient dimension
        current_dim_results = {'GLVQ': [], 'subspace_likelihood': []}
        pht_predictions_train = classifier.eval_pht_classifier(data_box_train_norm, data_box_train_norm, class_prior, np.identity(num_dim_ambient))
        pht_predictions_test = classifier.eval_pht_classifier(data_box_train_norm, data_box_test_norm, class_prior, np.identity(num_dim_ambient))

        for cost_func_type in ['GLVQ', 'subspace_likelihood']:
            current_dim_results[cost_func_type].append({'subspace_learned': np.identity(num_dim_ambient), 
                                                'pht_predictions_train': pht_predictions_train, 
                                                'pht_predictions_test': pht_predictions_test, 
                                                'ssl_time': 0})
        dimension_results[len(cols_vec)] = current_dim_results
    
        # record results of current fold
        outcomes_folds[fold_idx] = {'dimension_results': dimension_results,
                                    'train_index': train_index,
                                    'test_index': test_index
                                    }

        # --- save fold data ---
        
        # take time
        tend = time.time()
        total_time = tend - t0

        # save outcomes to file
        date_time_str = data_time_at_start.strftime("%Y_%m_%d_%H_%M_%S")
        save_path = path_to_project_root + '/data/processed/exp4a_' + date_time_str + '.pckl'    # determine path to Python/data/

        # create dict to collect save data
        save_data = {   
            'data_box_spherical': data_box_spherical,
            'exp4_opts': exp4a_opts, 
            'subspace_learn_opts': subspace_learn_opts,
            'outcomes_folds': outcomes_folds, 
            'folds_completed': fold_idx + 1,
            'total_time': total_time
            }

        # open a file, save data and close
        f = open(save_path, 'wb')
        pickle.dump(save_data, f)
        f.close()

def run_exp4b_gw_spherical_random_subsampling(exp4b_opts):

    '''
    Description
    ----------
    In this experiment, the PHT classifier and subspace learning are applied to a 3-class classification problem of Graviatational Waves.
    Spherical Gaussian Mixture models have been fitted to the posterior distributions.
    The subspace learners in use are subspace likelihood optimized with an iterative scheme and initialized randomly and from a run of GMLVQ.
    To estimate the out-of-sample performance, random subsampling is used. 
    The function automatically saves results to the 'data/processed/' folder and files names start with 'exp4a_'.
    
    Parameters
    ----------
    exp4b_opts : dict
        Dict containing the following parameters:

    num_splits : int
        positive int, number of folds for k-fold random subsampling

    num_rand_init : int
        positive int, number of random initializations per subspace learning

    test_size : float
        postive float, number between 0 and 1, percentage of data to be used for testing

    cols_vec : array
        np array int, indicates the dimensions of subspaces to search

    file_name_spherical : str
        name of file in data/interim/ to be loaded as density est. with spherical cov. matrices

    '''

    # start timer
    t0 = time.time()
    data_time_at_start = datetime.now()
    np.random.seed(0)                                                  

    # extract info from exp_opts
    num_rand_init = exp4b_opts['num_rand_init']
    cols_vec = exp4b_opts['cols_vec']
    num_splits = exp4b_opts['num_splits']
    test_size = exp4b_opts['test_size']

    # load data from previous density estimation
    file_name_spherical = exp4b_opts['file_name_spherical']
    data_box_spherical = DataBox()
    data_box_spherical = data_box_spherical.load_data('/interim/' + file_name_spherical)

    # find path to project root directory
    path_to_project_root = str(Path(__file__).parents[1])                                   

    # extract info from data box
    labels = data_box_spherical.get_labels()
    num_dim_ambient = data_box_spherical._dynamical_system.num_params
    num_examples = len(labels)

    # set general subspace learning options
    subspace_learn_opts = {
        'subspace_dim':                                  None,      # positive integer      
        'cost_function':                'subspace_likelihood',      # 'subspace_likelihood', 'GLVQ', 'subspace_likelihood_gradient'
        'ssl_approx':                                    None,      # boolean, only makes a different if 'subspace_likelihood' is selected and the density estimates are full-cov.
        'opt_mode':                               'iterative',      # 'batch', 'iterative'
        'init_type':                                     None,      # 'PCA', 'random'
        'class_prior':                                 'flat',      # 'empirical', 'flat'
        'scipy_min_method':                            'BFGS',      # 'BFGS'
        'scipy_min_tol':                                1e-04,      # positive float
        'scipy_min_maxiter':                              100,      # positive int
        'scipy_min_disp':                                True,      # boolean
    }

    # set density estimation options
    density_est_opts = {
        'type':             'bayesian_gaussian_mixture',
        'covariance_type':                  'spherical',    # 'full', 'diag' 'spherical' 
        'n_components':                              10,    # 10
        'n_init' :                                    5,    # 15
        'max_iter':                                 500,    # 200
        'trim_percent':                            0.99
    }

    # find class prior
    class_prior = set_class_prior(data_box_spherical.get_labels(), subspace_learn_opts['class_prior'])

    # define classifier
    classifier = LiMSClassifier(None)
    
    # set up k-fold cross-validation
    skf = StratifiedShuffleSplit(n_splits=num_splits, test_size=test_size)
    
    # create global variable to be passed to callback function
    global OPT_INFO
    OPT_INFO = config.OPT_INFO

    # loop over folds
    outcomes_folds = {}
    for fold_idx, (train_index, test_index) in enumerate(skf.split(np.zeros(num_examples), labels)):

        # update fold to be printed to screen
        OPT_INFO['fold'] = fold_idx + 1

        # initialize dict to hold results for current fold
        dimension_results = {}
        
        # create training and testing data sets
        data_box_train = data_box_spherical.select_examples(train_index)
        data_box_test = data_box_spherical.select_examples(test_index)

        # normalize training and test data
        data_box_train_norm, means, std_devs = data_box_train.shift_and_scale_data()       # apply standard-score transform
        data_box_test_norm, _, _ = data_box_test.shift_and_scale_data(means, std_devs)     # apply the same transform to test data

        # define density estimator object
        dens_est_train = DensityEstimator(data_box_train_norm, density_est_opts)
        dens_est_test = DensityEstimator(data_box_test_norm, density_est_opts)

        # run density estimation
        data_box_train_norm = dens_est_train.run_density_estimation()
        data_box_test_norm = dens_est_test.run_density_estimation()

        # loop over subspace dimensions
        for i in range(len(cols_vec)):

            # initialize dict to hold results for current dim
            current_dim_results = {'random_init': [], 'GMLVQ_init': []}

            # set the desired number of columns
            subspace_learn_opts['subspace_dim'] = cols_vec[i]

            # loop over random initializations                                                 # random initializations
            subspace_learn_opts['init_type'] = 'random'
            for trial in range(num_rand_init):
                
                # run subspace learning
                ssl = SubspaceLearner(data_box_train_norm, subspace_learn_opts)
                t0_ssl = time.time()
                db_subspace_likelihood = ssl.run_subspace_learning()
                tend_ssl = time.time()
                subspace_learned = db_subspace_likelihood.get_subspace_learned()
                pht_predictions_train = classifier.eval_pht_classifier(data_box_train_norm, data_box_train_norm, class_prior, subspace_learned['V_opt'])
                pht_predictions_test = classifier.eval_pht_classifier(data_box_train_norm, data_box_test_norm, class_prior, subspace_learned['V_opt'])

                # record results
                current_dim_results['random_init'].append({'subspace_learned': subspace_learned, 
                                                        'pht_predictions_train': pht_predictions_train, 
                                                        'pht_predictions_test': pht_predictions_test, 
                                                        'ssl_time': tend_ssl - t0_ssl})
            
            # loop over GMLVQ initializations                                                 # GMLVQ initializations
            subspace_learn_opts['init_type'] = 'GMLVQ'
            for trial in range(num_rand_init):
                
                # run subspace learning
                ssl = SubspaceLearner(data_box_train_norm, subspace_learn_opts)
                t0_ssl = time.time()
                db_subspace_likelihood = ssl.run_subspace_learning()
                tend_ssl = time.time()
                subspace_learned = db_subspace_likelihood.get_subspace_learned()
                pht_predictions_train = classifier.eval_pht_classifier(data_box_train_norm, data_box_train_norm, class_prior, subspace_learned['V_opt'])
                pht_predictions_test = classifier.eval_pht_classifier(data_box_train_norm, data_box_test_norm, class_prior, subspace_learned['V_opt'])

                # record results
                current_dim_results['GMLVQ_init'].append({'subspace_learned': subspace_learned, 
                                                        'pht_predictions_train': pht_predictions_train, 
                                                        'pht_predictions_test': pht_predictions_test, 
                                                        'ssl_time': tend_ssl - t0_ssl})

            # record results of current dimension
            dimension_results[i] = current_dim_results

        # record results for ambient dimension
        current_dim_results = {'random_init': [], 'GMLVQ_init': []}
        pht_predictions_train = classifier.eval_pht_classifier(data_box_train_norm, data_box_train_norm, class_prior, np.identity(num_dim_ambient))
        pht_predictions_test = classifier.eval_pht_classifier(data_box_train_norm, data_box_test_norm, class_prior, np.identity(num_dim_ambient))

        for init_type in ['random_init', 'GMLVQ_init']:
            current_dim_results[init_type].append({'subspace_learned': np.identity(num_dim_ambient), 
                                                'pht_predictions_train': pht_predictions_train, 
                                                'pht_predictions_test': pht_predictions_test, 
                                                'ssl_time': 0})
        dimension_results[len(cols_vec)] = current_dim_results
    
        # record results of current fold
        outcomes_folds[fold_idx] = {'dimension_results': dimension_results,
                                    'train_index': train_index,
                                    'test_index': test_index
                                    }

        # --- save fold data ---
        
        # take time
        tend = time.time()
        total_time = tend - t0

        # save outcomes to file
        date_time_str = data_time_at_start.strftime("%Y_%m_%d_%H_%M_%S")
        save_path = path_to_project_root + '/data/processed/exp4b_' + date_time_str + '.pckl'    # determine path to Python/data/

        # create dict to collect save data
        save_data = {   
            'data_box_spherical': data_box_spherical,
            'exp4b_opts': exp4b_opts, 
            'subspace_learn_opts': subspace_learn_opts,
            'outcomes_folds': outcomes_folds, 
            'folds_completed': fold_idx + 1,
            'total_time': total_time
            }

        # open a file, save data and close
        f = open(save_path, 'wb')
        pickle.dump(save_data, f)
        f.close()

def run_exp5_pred_worst_case_inits(exp5_opts):
    
    '''
    Description
    ----------
    In this experiment, we consider subspace learning for the prednisone spiral problem. 
    The subspace learning is initialized with a 2D subspace which is orthogonal to the subspace used for generating the data.
    The history of optimization is recorded and a few quantities are then calculated as a function of the number of iterations.
    The quantities are a) Grassmann-distance to the generating subspace, b) loglikelihood value, c-d) classification performance on the training and test sets.
    The function automatically saves results to the 'data/processed/' folder and files names start with 'exp5_'.
    
    Parameters
    ----------
    exp5_opts : dict
        Dict containing the following parameters:

    num_orth_init : int
        positive int, number of random orthogonal initializationis for subspace learning

    file_name_spherical : str
        name of file in data/interim/ to be loaded as density est. with spherical cov. matrices

    '''

    # start timer and set seed
    t0 = time.time()
    np.random.seed(0)

    # extract info from exp_opts
    num_orth_init = exp5_opts['num_orth_init']

    # load data from previous density estimation
    file_name_spherical = exp5_opts['file_name_spherical']
    data_box_spherical = DataBox()
    data_box_spherical = data_box_spherical.load_data('/interim/' + file_name_spherical)

    # extract number of parameters
    num_dim_ambient = data_box_spherical._dynamical_system.num_params
    ds_name = data_box_spherical._dynamical_system.name

    # make 50/50 train and test split
    labels = data_box_spherical.get_labels()
    skf = StratifiedKFold(n_splits = 2, shuffle=True, random_state=0)
    split1, split2 = skf.split(np.zeros(len(labels)), labels)
    train_index = split1[0]
    test_index = split1[1]
    data_box_spherical_train = data_box_spherical.select_examples(train_index)
    data_box_spherical_test = data_box_spherical.select_examples(test_index)
    data_box_train_norm, means, std_devs = data_box_spherical_train.shift_and_scale_data()       # apply standard-score transform
    data_box_test_norm, _, _ = data_box_spherical_test.shift_and_scale_data(means, std_devs)     # apply the same transform to test data

    # set general subspace learning options
    subspace_learn_opts = {
        'subspace_dim':                                     2,      # positive integer      
        'cost_function':                'subspace_likelihood',      # 'subspace_likelihood', 'GLVQ', 'subspace_likelihood_gradient'
        'ssl_approx':                                    None,      # boolean, only makes a different if 'subspace_likelihood' is selected and the density estimates are full-cov.
        'opt_mode':                                   'batch',      # 'batch', 'iterative'
        'init_type':                                     None,      # 'PCA', 'random'
        'class_prior':                                 'flat',      # 'empirical', 'flat'
        'scipy_min_method':                            'BFGS',      # 'BFGS'
        'scipy_min_tol':                                1e-05,      # positive float
        'scipy_min_maxiter':                              100,      # positive int
        'scipy_min_disp':                                True,      # boolean
    }

    # define classifier
    classifier = LiMSClassifier(None)

    # find class prior
    class_prior = set_class_prior(data_box_spherical.get_labels(), subspace_learn_opts['class_prior'])

    # extract generating subuspace and find normal vector
    subspace_true = data_box_spherical.get_subspace()
    v1 = subspace_true['v1']
    v2 = subspace_true['v2']
    V_true = np.vstack((v1, v2)).T
    normal_vec = np.cross(v1, v2)

    # initialize outcomes dict
    outcomes = {}

    # loop over num_orth_init orthogonal subspaces
    for i in range(num_orth_init):

        # adjust initial subspace to be orthogonal to the generating subspace
        rand_vec = np.dot(V_true, np.random.rand(2))
        rand_vec /= np.linalg.norm(rand_vec)
        subspace_learn_opts['init_type'] = np.vstack((normal_vec, rand_vec)).T

        # run subspace learning
        ssl = SubspaceLearner(data_box_train_norm, subspace_learn_opts)
        db_spherical = ssl.run_subspace_learning()
        subspace_learned = db_spherical.get_subspace_learned()

        # loop over optimization history
        predictions = {}
        num_opt_steps = len(subspace_learned['opt_history']['fun'])
        
        # predictions for V0
        pht_predictions_train = classifier.eval_pht_classifier(data_box_train_norm, data_box_train_norm, class_prior, subspace_learned['V0'])
        pht_predictions_test = classifier.eval_pht_classifier(data_box_train_norm, data_box_test_norm, class_prior, subspace_learned['V0'])
        predictions[0] = {
            'train': pht_predictions_train,
            'test': pht_predictions_test
        }
        
        # loop over remaining iterations
        for j in range(num_opt_steps):

            # evaluate pht-classifier and store predictions
            pht_predictions_train = classifier.eval_pht_classifier(data_box_train_norm, data_box_train_norm, class_prior, subspace_learned['opt_history']['V'][j])
            pht_predictions_test = classifier.eval_pht_classifier(data_box_train_norm, data_box_test_norm, class_prior, subspace_learned['opt_history']['V'][j])
            predictions[j + 1] = {
                'train': pht_predictions_train,
                'test': pht_predictions_test
            }

        # update outcomes dict
        outcomes[i] = {
            'subspace_learned': subspace_learned,
            'predictions': predictions
        }

    # stop timer
    tend = time.time()
    total_time = tend - t0

    # save outcomes to file
    path_to_project_root = str(Path(__file__).parents[1])                                   # find path to project root directory
    current_date_time = datetime.now()                                                      # find current time and date
    date_time_str = current_date_time.strftime("%Y_%m_%d_%H_%M_%S")
    save_path = path_to_project_root + '/data/processed/exp5_' + date_time_str + '_' + ds_name + '.pckl'    # determine path to Python/data/

    # create dict to collect save data
    save_data = {   
        'data_box_spherical': data_box_spherical,
        'train_index': train_index,
        'test_index': test_index,
        'exp5_opts': exp5_opts, 
        'subspace_learn_opts': subspace_learn_opts,
        'outcomes': outcomes, 
        'total_time': total_time
        }

    # open a file, save data and close
    f = open(save_path, 'wb')
    pickle.dump(save_data, f)
    f.close()

def run_exp6_pred_GMLVQ_inits(exp6_opts):
    
    '''
    Description
    ----------
    In this experiment, we consider subspace learning for the prednisone spiral problem.
    This expeirments aims to investigate the question: How much (if at all) does the PHT classifier learn on top of the GMLVQ classifier?
    For a number of random initialization, the following happens:
    a) the GMLVQ classifier is trained and then evaluated on the test data. The two dominant relevance-directions are recored as V_GMLVQ.
    b) subspace learning for the PHT classifier is initialized from V_GMLVQ. The PHT classifier is evaluated on the test-data for V_GMLVQ and V_opt.
    The function automatically saves results to the 'data/processed/' folder and files names start with 'exp6_'.
    
    Parameters
    ----------
    exp6_opts : dict
        Dict containing the following parameters:

    num_rand_init : int
        positive int, number of random initializationis for subspace learning

    file_name_spherical : str
        name of file in data/interim/ to be loaded as density est. with spherical cov. matrices

    '''

    # start timer and set seed
    t0 = time.time()
    np.random.seed(0)

    # load data from previous density estimation
    file_name_spherical = exp6_opts['file_name_spherical']
    data_box_spherical = DataBox()
    data_box_spherical = data_box_spherical.load_data('/interim/' + file_name_spherical)

    # extract number of parameters
    num_rand_init = exp6_opts['num_rand_init']
    ds_name = data_box_spherical._dynamical_system.name

    # make 50/50 train and test split, normalize
    labels = data_box_spherical.get_labels()
    skf = StratifiedKFold(n_splits = 2, shuffle=True, random_state=0)
    split1, split2 = skf.split(np.zeros(len(labels)), labels)
    train_index = split1[0]
    test_index = split1[1]
    data_box_spherical_train = data_box_spherical.select_examples(train_index)
    data_box_spherical_test = data_box_spherical.select_examples(test_index)
    data_box_train_norm, means, std_devs = data_box_spherical_train.shift_and_scale_data()       # apply standard-score transform
    data_box_test_norm, _, _ = data_box_spherical_test.shift_and_scale_data(means, std_devs)     # apply the same transform to test data

    # set general subspace learning options
    subspace_learn_opts = {
        'subspace_dim':                                     2,      # positive integer      
        'cost_function':                'subspace_likelihood',      # 'subspace_likelihood', 'GLVQ', 'subspace_likelihood_gradient'
        'ssl_approx':                                    None,      # boolean, only makes a different if 'subspace_likelihood' is selected and the density estimates are full-cov.
        'opt_mode':                                   'batch',      # 'batch', 'iterative'
        'init_type':                                     None,      # 'PCA', 'random'
        'class_prior':                                 'flat',      # 'empirical', 'flat'
        'scipy_min_method':                            'BFGS',      # 'BFGS'
        'scipy_min_tol':                                1e-05,      # positive float
        'scipy_min_maxiter':                              100,      # positive int
        'scipy_min_disp':                                True,      # boolean
    }

    # set class prior
    class_prior = set_class_prior(data_box_train_norm.get_labels(), subspace_learn_opts['class_prior'])

    # define PHT classifier
    classifier = LiMSClassifier(None)

    # merge all means from all mixtures together
    means_merged_train, labels_merged_train = data_box_train_norm.merge_means_and_labels()
    means_merged_test, labels_merged_test = data_box_test_norm.merge_means_and_labels()

    # initialize outcomes dict
    outcomes = {}

    for i in range(num_rand_init):

        # create model object to fit data to
        model = GMLVQ(
            distance_type="adaptive-squared-euclidean",
            activation_type="sigmoid",
            activation_params={"beta": 1},
            solver_type="steepest-gradient-descent",
            solver_params={"max_runs": 100, "batch_size": 1, "step_size": 0.1},
            prototype_n_per_class = 5,
            random_state = i,
        )

        # train the model and extract relevance matrix
        model.fit(means_merged_train, labels_merged_train)
        relevance_matrix = model.lambda_

        # Predict the labels using the trained model
        GMLVQ_predictions_train_crisp = model.predict(means_merged_train)
        GMLVQ_predictions_test_crisp = model.predict(means_merged_test)
        GMLVQ_predictions_train = np.zeros((len(GMLVQ_predictions_train_crisp), 2))
        GMLVQ_predictions_test = np.zeros((len(GMLVQ_predictions_test_crisp), 2))
        for k in range(len(GMLVQ_predictions_train_crisp)):
            GMLVQ_predictions_train[k, int(GMLVQ_predictions_train_crisp[k])] = 1.
        for k in range(len(GMLVQ_predictions_test_crisp)):
            GMLVQ_predictions_test[k, int(GMLVQ_predictions_test_crisp[k])] = 1.

        # perform SVD on relevance_matrix
        U, S, Vh = np.linalg.svd(relevance_matrix, full_matrices=False)

        # extract dominant subspace_dim directions of relevance_matrix
        V_GMLVQ = np.copy(U[:, 0:subspace_learn_opts['subspace_dim']])

        # run subspace learning
        subspace_learn_opts['init_type'] = V_GMLVQ
        ssl = SubspaceLearner(data_box_train_norm, subspace_learn_opts)
        db_spherical = ssl.run_subspace_learning()
        subspace_learned = db_spherical.get_subspace_learned()

        # evaluate PHT predictions on subspace learned by GMLVQ
        pht_predictions_train_GMLVQ = classifier.eval_pht_classifier(data_box_train_norm, data_box_train_norm, class_prior, V_GMLVQ)
        pht_predictions_test_GMLVQ = classifier.eval_pht_classifier(data_box_train_norm, data_box_test_norm, class_prior, V_GMLVQ)

        # evaluate PHT predictions on subspace learned by via our method
        pht_predictions_train = classifier.eval_pht_classifier(data_box_train_norm, data_box_train_norm, class_prior, subspace_learned['V_opt'])
        pht_predictions_test = classifier.eval_pht_classifier(data_box_train_norm, data_box_test_norm, class_prior, subspace_learned['V_opt'])

        # store predictions in outcomes
        outcomes[i] = {
            'GMLVQ_labels_train':           labels_merged_train,
            'GMLVQ_labels_test':            labels_merged_test,
            'GMLVQ_predictions_train':      GMLVQ_predictions_train,
            'GMLVQ_predictions_test':       GMLVQ_predictions_test,
            'pht_predictions_train_GMLVQ':  pht_predictions_train_GMLVQ,
            'pht_predictions_test_GMLVQ':   pht_predictions_test_GMLVQ,
            'pht_predictions_train':        pht_predictions_train,
            'pht_predictions_test':         pht_predictions_test
        }

    # evaluate PHT predictions on ambient space
    pht_predictions_train = classifier.eval_pht_classifier(data_box_train_norm, data_box_train_norm, class_prior, np.identity(3))
    pht_predictions_test = classifier.eval_pht_classifier(data_box_train_norm, data_box_test_norm, class_prior, np.identity(3))
    outcomes[num_rand_init] = {
        'pht_predictions_train':        pht_predictions_train,
        'pht_predictions_test':         pht_predictions_test
    }

    # stop timer
    tend = time.time()
    total_time = tend - t0

    # save outcomes to file
    path_to_project_root = str(Path(__file__).parents[1])                                   # find path to project root directory
    current_date_time = datetime.now()                                                      # find current time and date
    date_time_str = current_date_time.strftime("%Y_%m_%d_%H_%M_%S")
    save_path = path_to_project_root + '/data/processed/exp6_' + date_time_str + '_' + ds_name + '.pckl'    # determine path to Python/data/

    # create dict to collect save data
    save_data = {   
        'data_box_spherical': data_box_spherical,
        'train_index': train_index,
        'test_index': test_index,
        'exp6_opts': exp6_opts, 
        'subspace_learn_opts': subspace_learn_opts,
        'outcomes': outcomes, 
        'total_time': total_time
        }

    # open a file, save data and close
    f = open(save_path, 'wb')
    pickle.dump(save_data, f)
    f.close()