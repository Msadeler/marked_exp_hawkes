# Simulation, estimation  and test for a Multidimensional Marked Exponential Hawkes Process ( MMEHP )

## Introduction

This project provides a Python implementation for simulating and estimating a uni or multi-dimensional marked Hawkes process. In addition to simulation and estimation features, the code includes testing procedures to verify the model fitting to observed data.

## Features

### 1. Simulation of a Multidimensional Marked Hawkes Process
- **Process simulation:** Two main class allow for simulating of a marked Hawkes process, ``exp_thinning_hawkes_marked``and ``multivariate_exponential_hawkes_marked`` based on the thinning procedure. Users must specify model parameters, such as the baseline intensities, the interaction coefficients, the caracteritic times and mark distributions.

### 2. Model Parameter Estimation
- **Likelihood Calculation Functions:** Different functions such as ``multivariate_marked_likelihood``, ``loglikelihoodMarkedHawkes``,``multivariate_loglikelihood_simplified`` or ``loglikelihood``, allow to cumpute the likelihood of a parameter in different Hawkes model, given a set of data.
- **Parameter Estimation:** Two class allows to compute the parameter of either a multidimensional or unidimensional Hawkes process when one or more repetition of the process are available: ``estimator_unidim_multi_rep``, ``estimator_multidim_multi_rep``, ``loglikelihood_estimator`` and ``multivariate_estimator` . All of them require the user so specify the model characteristic, such as the type of parametrisation chosen for the impact function and the density of the mark, the associated name of paramter and the space they are embedded in.   

### 3. Testing Procedures for Model Fit
- **Testing a Specific Coefficient Value:** This procedure allows testing whether a given coefficient in the model equals a specific predefined value.
- **Testing Coefficient Equality:** This procedure allows testing whether two or more coefficients in the model are equal.
- **Goodness-of-Fit (GOF) Tests:** These tests check whether the observed data is consistent with a Hawkes process, such as evaluating the uniform distribution of aggregated points or the exponential distribution of process increments.


## Usage 

``py
from functions.multivariate_exponential_process import *
from functions.paramtrised_function import *
import scipy 
import numpy as np

np.rando.seed(0)

# Define model parameters

m, a, b = 1, -1, 2
Tmax = 5

# Simulate the process

hawkes_multi = multivariate_exponential_hawkes_marked(m=m,
                                                      a=a, 
                                                      b=b, 
                                                      phi = phi, 
                                                      F= F, 
                                                      arg_phi={'gamma':phi_arg}, 
                                                      arg_F={'psi': 2}, 
                                                      max_jumps  = 10)

hawkes_multi.simulate()

fig,ax = plt.subplots(2,2, figsize = (10,10))
hawkes_multi.plot_intensity(ax = ax)
``
<img src="./plot/simulation_MMEHP.png" width="500">