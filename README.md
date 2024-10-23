# General Description
This code library contains the experimental code which was used to produce the results presented in the manuscript "Discriminative subspace learning for model- and data-driven discovery".

---

# Installation

1. Pull repository
2. Setup a virtual environment (optional)
3. Install packaged specified in "requirements.txt": ```pip install -r requirements.txt``` 
4. Install code base as package: ```pip install -e .```

---

# Demo File
The notbook ```experiments/demo.ipynb``` contains a brief demonstration of the code functionality.

---

# Reproduction of Manuscript Figures
The notebooks ```experiments/prednisone_postprocessing.ipynb``` and ```experiments/gravitational_waves_postprocessing.ipynb``` can be used to reproduce all figures presented in "Discriminative subspace learning for model- and data-driven discovery".

---

# General Information

## Directory Structure

1. ```data/``` is the location where all external, interim and processed data is stored.
2. ```experiments/``` contains all experimental code associated with the experiments discussed in the manuscript.
3. ```src/``` contains all  source code related to the handling of dynamical systems, sampling from posterioirs, density estimation, the LiMS classifier and subspace learning.


## Class Descriptions
The core classes are contained in the folder ```src/```. Here is a brief description of each.

### DynamicalSystem (dynamical_systemn.py)
This class contains the core functionality of the dynamical system to be studied. To define a new system, a sub-class which inherits from dynamical system is to be created in the "src/dynamical_systems" folder (see e.g. Prednisone3D).
Once a dynamical system is define, the class methods allow easy access to function solving the associated ODEs.

### Sampler (sampler.py)
This class contains the interface to the different samplers in use (e.g. dynesty).

### DensityEstimator (density_estimator.py)
This class contains the interface to the different density estimation methods in use (e.g. sklearn's BayesianGaussianMixture).

### SubspaceLearner (subspace_learner.py)
This class contains the all functionality related to LiMS subspace learning.

### DataBox (data_box.py)
The DataBox class contains the timeseries data, posterior samples and all other useful intermediate data processing steps.

---

## Workflow

The suggested workflow is the following

1) define a dynamical system and generate or load time series data, create a new DataBox object, add the data and save
2) call the sampler to find the posterior distribution of each time series observation given the dynamical model, store again in the same DataBox and save
3) call the density estimator to fit a Gaussian mixture to the posterior samples, store in the same DataBox and save
4) call the subspace learner to learn discriminative subspaces from the approximated posterior distributions, store the results in the same DataBox and save

At each step of the data processing pipeline, the intermediate data is stored in the DataBox and can be saved to the ```data/interim/```
Steps 1) tp 4) do not need to be carried out directly after each other every time.

---

# References and Acknowledgements
The code in this repository is utilizing functionality from the following packages

- dynesty: A Dynamic Nested Sampling package for computing Bayesian posteriors and evidences. Version 2.1.3. Accessed on October 18, 2024. Available at [dynesty](https://github.com/joshspeagle/dynesty)

- scikit-learn: Machine Learning in Python. Version 1.4.1. Accessed on October 18, 2024. Available at [scikit-learn](https://scikit-learn.org/1.5/index.html)

- corner: Make some beautiful corner plots. Version 2.2.2. Accessed on October 18, 2024. Available at [corner](https://corner.readthedocs.io/en/latest/#)
