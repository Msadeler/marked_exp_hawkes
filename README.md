# Simulation, estimation, and testing for a multidimensional marked exponential Hawkes process (MMEHP)

## Introduction

This project provides a Python implementation for simulating, estimating, and testing both unidimensional and multidimensional marked Hawkes processes. In addition to simulation and estimation features, the code includes testing procedures to verify the model's fit to observed data.

## Features

### 1. Simulation of a Multidimensional Marked Hawkes Process
- **Process Simulation:** Two main classes, `exp_thinning_hawkes_marked` and `multivariate_exponential_hawkes_marked`, allow for the simulation of marked Hawkes processes based on the thinning procedure. Users must specify model parameters such as baseline intensities, interaction coefficients, characteristic times, and mark distributions.

### 2. Model Parameter Estimation
- **Likelihood Calculation Functions:** Various functions such as `multivariate_marked_likelihood`, `loglikelihoodMarkedHawkes`, `multivariate_loglikelihood_simplified`, and `loglikelihood` allow for the computation of the likelihood of parameters in different Hawkes models, given a dataset.
- **Parameter Estimation:** Two classes, `estimator_unidim_multi_rep` and `estimator_multidim_multi_rep`, are provided to estimate parameters for either unidimensional or multidimensional Hawkes processes when one or more repetitions of the process are available. These classes require the user to specify model characteristics, such as the type of parameterization chosen for the impact function and the density of the marks, along with the associated parameter names and the space in which they are embedded.

### 3. Testing Procedures
- **Testing a Specific Coefficient Value:** This procedure, associated with the estimator classes when multiple repetitions are available, allows testing whether a given coefficient in the model equals a specific predefined value.
- **Testing Coefficient Equality:** This procedure, also associated with the estimator classes when multiple repetitions are available, allows testing whether two coefficients in the model are equal.
- **Goodness-of-Fit (GOF) Tests:** This procedure, associated with the estimator classes when multiple repetitions are available, allows testing the model's fit to the data.

## Usage

### Simulation

```python
from functions.multivariate_exponential_process import *
from functions.parametrised_function import *
import scipy 
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Define model parameters
m, a, b = 1, -1, 2
phi = None  # Define your function phi
F = None  # Define your function F
phi_arg = {'gamma': 0.5}  # Example argument for phi

# Simulate the process
hawkes_multi = multivariate_exponential_hawkes_marked(
    m=m,
    a=a, 
    b=b, 
    phi=phi, 
    F=F, 
    arg_phi=phi_arg, 
    arg_F={'psi': 2}, 
    max_jumps=10
)

hawkes_multi.simulate()
timestamps = hawkes_multi.timestamps
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
hawkes_multi.plot_intensity(ax=ax)
```

<img src="./plot/simulation_MMEHP.png" width="500">



### Estimation

```py 
np.random.seed(0)

# Simulation 
m = 1
a = -1
b = 1

hawkes_multi = exp_thinning_hawkes_multi_marked(
    m=m,
    a=a, 
    b=b, 
    n=200,
    max_jumps=500
)
hawkes_multi.simulate()

# Estimation
estimator = estimator_unidim_multi_rep(a_bound=None, bound_b=None)
estimator.fit(hawkes_multi.timeList, max_jump=True)

```

### Test

```py 

## test on the value of the coefficient
stat = learner_hawkes.test_one_coeff( coefficient_index=0, value = 1,plot=True)

## GOF procedure with bootstrap
stat_hawkes = learner_hawkes.GOF_bootstrap(test_type = 'uniform', Nb_SubSample=100, plot = True)

```


## Examples 

Complete usage examples are available in the notebook ``example.ipynb``, with scripts illustrating simulation, estimation, and testing.


## Dependencies

This code was implemented using Python 3.12.4 and needs Numpy, Matplotlib, Scipy, rpy2, IPython, functools, multiprocessing.
