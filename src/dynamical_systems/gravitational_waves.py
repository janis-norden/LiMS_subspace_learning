from ..dynamical_system import DynamicalSystem
from ..data_box import DataBox
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

class GravitationalWaves(DynamicalSystem):

    ### attributes ###
    name = 'gravitational_waves'                            # string with name of the system
    ROI = None                                              
    num_states = None                                       # number of state variables in the model
    num_params = 4                                          # number of parameters in the model

    ### methods: polymorphisms ###

    def __init__(self):
        super().__init__()

    ### methods: model-specific ###
    def load_data(self):

        ''' load gravitational wave data from data/external/ '''

        path_to_project_root = str(Path(__file__).parents[2])

        # load the labels from csv
        with open(path_to_project_root + '/data/external/gravitational_waves/Labels.csv', 'r') as f:
            reader = csv.reader(f)
            labels_in = list(reader)
            labels = np.array(labels_in[0], dtype = int) - 1

        # load posteriors from csv
        posteriors_dict = {}

        for i in range(117):

            # adjust file name
            file_name = path_to_project_root + '/data/external/gravitational_waves/Posterior_n' + str(i + 1).zfill(3) + '.csv'

            # load the posterior from csv
            with open(file_name, 'r') as f:
                reader = csv.reader(f)
                posterior_in = list(reader)
            
            posterior = {}
            posterior['samples'] = np.array(posterior_in, dtype = float)
            posterior['weights'] = np.ones(len(posterior_in)) / len(posterior_in)

            posteriors_dict[i] = posterior

        # create data box and add labels and posterior samples
        data_box = DataBox()

        data_box.set_labels(labels)
        data_box.set_posterior_samples(posteriors_dict)
        data_box.set_dynamical_system(self)

        return data_box