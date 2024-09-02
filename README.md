# Test for Exponential Hawkes Process

The repo contains different class for simulating multidimensional marked exponential Hawkes processes, along with functions to compute the likelihood and the MLE estimator of the process, using a dataset.


## Example

We present two class, allowing for simulation and estimation of a marked hawkes process, named ```multivariate_exponential_hawkes_marked``` and ```estimator_unidim_multi_rep```.

```py

from functions.multivariate_exponential_process import *
import scipy 
from function_test import *


if __name__ == "__main__":

    # Set seed
    np.random.seed(0)

    m = np.array([0.5, 0.2]).reshape((2,1))
    a = np.array([[0.4, 0.2,], 
                    [-0.4, 0.3]] )
    b = np.array([[1],[1.5]])


    Tmax = 10
    phi_arg = 0.5
        
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

```
<img src="./plot/simulation_MMEHP.png" width="500">

