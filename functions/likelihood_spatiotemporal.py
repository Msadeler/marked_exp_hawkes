# Imports
import numpy as np
from scipy.optimize import minimize
from functions.kernel_spatial_component import *

def likelihood_spatio_temp_gaussian_kernel(theta, pc, radius, tList, list_locak):

    
    """
    Computation of the loglikelihood for an spatiotemporal exponential Hawkes process in self-exciting cases. 
    Estimation for a single realization.
    
    Parameters
    ----------
    theta : tuple of float
        Tuple containing the parameters to use for estimation.

    pc: array of shape (2,1)
        Localisation of the place field where intensity is maximale
    radius: positive float
        characteristic radius of the place field
    tList : list of float
        List containing all the lists of data (event times).

    Returns
    -------
    likelihood : float
        Value of likelihood, either for 1 realization or for a batch. 
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """


    mu_pn, mu_npc, alpha, beta = theta 
    return()


def likelihood_spatio_temp_indic_kernel(theta, pc, radius, tList, xList):

    
    """
    Computation of the loglikelihood for an spatiotemporal exponential Hawkes process with indic value for the baseline intensity. 
    Estimation for a single realization.
    
    Parameters
    ----------
    theta : tuple of float
        Tuple containing the parameters to use for estimation.

    pc: array of shape (2,1)
        Localisation of the place field where intensity is maximale
    radius: positive float
        characteristic radius of the place field
    tList : list of float
        List containing all the lists of data (event times).

    Returns
    -------
    likelihood : float
        Value of likelihood, either for 1 realization or for a batch. 
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """

    print(theta)
    mu_pc, mu_npc, alpha, beta = theta 

    ## initialisation 
    if mu_pc <= 0 or mu_npc <= 0 or beta <= 0:
        return(1e10)

    else: 
        v_ball = np.pi*radius**2
        v_tot = np.pi

        compensator_k = (mu_npc*(1-v_ball) + mu_pc*v_ball)*tList[1] ## value of the compensator between Tk and Tk+1

        mu_k = indic_kernel(xList[1]-pc, mu_pc, mu_npc, radius)[0] ## value of mu(Xk)
        lambda_k = mu_k

        likelihood = np.log(lambda_k)- compensator_k
        previous_time = tList[2]


        for t,x in zip(tList[2:], xList[2:]):

            lambda_k_after = lambda_k + alpha - mu_k
            aux_compensator = (mu_pc*v_ball +mu_npc*(1- v_ball) )*( t- previous_time)+ 2*v_tot*(1- np.exp(-beta*(t-previous_time)))*(lambda_k_after)/beta
            
            if lambda_k_after<= 0:  
                compensator_k = aux_compensator + (1-v_ball)*(-lambda_k_after/ mu_npc - mu_npc*np.log(-lambda_k_after/mu_npc)-1) + v_ball*(-lambda_k_after/mu_pc - mu_pc*np.log(-lambda_k_after/mu_pc)-1)

            else: 
                compensator_k = aux_compensator

            mu_k = indic_kernel(x-pc, mu_pc=mu_pc, mu_npc=mu_npc, r=radius)[0]
            lambda_k = mu_k + np.exp(-beta*(t-previous_time))*lambda_k_after

            likelihood+= np.log(lambda_k)- compensator_k
    

        return(likelihood- np.log(lambda_k))


