# Imports
import numpy as np
from scipy.optimize import minimize


def likelihood_Poisson(theta, tList, **kwargs):
     
    """
    Exact computation of the loglikelihood for an Poisson Process. 
    Estimation for a single realization.
    
    Parameters
    ----------
    theta : tuple of float
        Tuple containing the parameters to use for estimation.
    tList : list of float
        List containing all the lists of data (event times).

    Returns
    -------
    likelihood : float
        Value of likelihood, either for 1 realization or for a batch. 
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.

    """
    if isinstance(theta, (np.ndarray,list)):
        mu = theta[0]  
    elif isinstance(theta, float):
        mu = theta  

    loglik = (len(tList)-2)*np.log(mu) - (tList[-1]- tList[0])*mu

    return(-loglik)

 

def likelihood_daichan(a, mu, b, tList):


    compensator_k = mu * tList[1]
    lambda_avant = mu
    lambda_k = mu + a

    if lambda_avant <= 0:
        return 1e5

    likelihood = np.log(lambda_avant) - compensator_k

    # Iteration
    for k in range(2, len(tList)-1):

        if lambda_k >= 0:
            C_k = lambda_k - mu
            tau_star = tList[k] - tList[k - 1]
        else:
            C_k = -mu
            tau_star = tList[k] - tList[k - 1] - (np.log(-(lambda_k - mu)) - np.log(mu)) / b

        lambda_avant = mu + (lambda_k - mu) * np.exp(-b * (tList[k] - tList[k - 1]))
        lambda_k = lambda_avant + a
        compensator_k = mu * tau_star + (C_k / b) * (1 - np.exp(-b * tau_star))

        if lambda_avant <= 0:
            return 1e5
        
        

        likelihood += np.log(lambda_avant) - compensator_k
    
    k = len(tList)-1
    
    if lambda_k >= 0:
        C_k = lambda_k - mu
        tau_star = tList[k] - tList[k - 1]
    else:
        C_k = -mu
        tau_star = tList[k] - tList[k - 1] - (np.log(-(lambda_k - mu)) - np.log(mu)) / b

    compensator_k = mu * tau_star + (C_k / b) * (1 - np.exp(-b * tau_star))

    likelihood -=  compensator_k
    # We return the opposite of the likelihood in order to use minimization packages.
    
    return -likelihood



def loglikelihoodMarkedHawkes(x, tList,  name_arg_f, name_arg_phi, f, phi):
    
    """
    Exact computation of the loglikelihood for an exponential marked Hawkes process. 
    Estimation for a single realization.
    
    Parameters
    ----------
    theta : tuple of float
        Tuple containing the parameters to use for estimation.
        
    tList : list of float
        List containing all the lists of data (event times).
        
    markeList: list of float 
        List containing the value of the mark at the time of each jump int tList
        
    f: density function
        The density of the mark , the density must have at list two paramaters: the mark value (first argment) and the time value (second one)
        
    phi: function 
        The function describing the impact of the mark on the interaction between each neuron
    
    arg_phi: parameters for the function phi
        dictionnary of parameters allowing to compute the function phi,  this dictionnary does not containg the argument mark 
    
    arg_f: parameters for the density f
        dictionnary of parameters allowing to compute the density, this dictionnary does not containg the argument mark or time


    Returns
    -------
    likelihood : float
        Value of likelihood 
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """
    arg_f = dict(zip(list(name_arg_f) ,x[:len(name_arg_f)]))
    arg_phi = dict(zip(list(name_arg_phi),x[len(name_arg_f) :len(name_arg_f)+len(name_arg_phi)]))
    theta = x[len(name_arg_f)+len(name_arg_phi):]
    

    # unpack hawkes parameters 
    lambda0, a, b  = theta

    if lambda0 <= 0 or b <= 0:
        return 1e5
    
    else: 
        compensator_k = lambda0 * (tList[1][0]-tList[0][0])
        lambda_avant = lambda0
        lambda_k = lambda0 + a*phi(tList[1][1],**arg_phi, **arg_f)
    
    likelihood = np.log(lambda_avant) - compensator_k 
    
    
    for k in range(2, len(tList)-1):
        
        if lambda_k >= 0:            
            C_k = lambda_k - lambda0
            tau_star = tList[k][0] - tList[k - 1][0]
        else:
            C_k = -lambda0
            tau_star = tList[k][0] - tList[k - 1][0] - (np.log(-(lambda_k - lambda0)) - np.log(lambda0)) / b

        lambda_avant = lambda0 + (lambda_k - lambda0) * np.exp(-b * (tList[k][0] - tList[k - 1][0]))
        lambda_k = lambda_avant + a*phi(tList[k][1],**arg_phi, **arg_f)
        compensator_k = lambda0 * tau_star + (C_k / b) * (1 - np.exp(-b * tau_star))

        if lambda_avant <= 0:
             return 1e5
         
        likelihood += np.log(lambda_avant) - compensator_k  
    

    if lambda_k >= 0:
        C_k = lambda_k - lambda0
        tau_star = tList[k][0] - tList[k - 1][0]
    else:
        C_k = -lambda0
        tau_star = tList[k][0] - tList[k - 1][0] - (np.log(-(lambda_k - lambda0)) - np.log(lambda0)) / b

    compensator_k = lambda0 * tau_star + (C_k / b) * (1 - np.exp(-b * tau_star))

    if lambda_avant <= 0:
        return 1e5

    likelihood -= compensator_k
    
    likelihood_mark = np.sum([np.log(f(mark , **arg_f)) for time, mark in tList[1:-1]])
    likelihood += likelihood_mark
    
        
     # We return the opposite of the likelihood in order to use minimization packages.
    return -likelihood


def minimization_unidim_unmark(list_times, loss, initial_guess, bounds, options):
     return(minimize(loss, x0 = initial_guess, method="L-BFGS-B",args=(list_times), bounds=bounds, options=options))

def minimization_unidim_marked(tlist,loss, initial_guess,f,phi, name_arg_f, name_arg_phi, bounds, options):
    return(minimize(loss,x0=initial_guess, method="L-BFGS-B",args=(tlist, name_arg_f,name_arg_phi,f,phi), bounds=bounds, options=options))


def minimization_multidim_marked(tlist,loss, initial_guess,f,phi, name_arg_f, name_arg_phi,bounds, options):
    return(minimize(loss,initial_guess, method="L-BFGS-B",args=(tlist,name_arg_f,name_arg_phi,f,phi), bounds=bounds, options=options))

def minimization_multidim_unmark(list_times :  list, loss, initial_guess , bounds, options, dim):
     return(minimize(loss, initial_guess, method="L-BFGS-B",args=(list_times, dim), bounds=bounds, options=options))


def multivariate_marked_likelihood(x:list, tList: list, name_arg_phi: list, name_arg_f,phi ,f, nb_arg_phi, dim=None, dimensional=False):
    """
    Parameters
    ----------
    theta : tuple of array
        Tuple containing 3 arrays. First corresponds to vector of baseline intensities mu. Second is a square matrix
        corresponding to interaction matrix a. Last is vector of recovery rates b.

    tList : list of tuple
        List containing tuples (t, d, m) where t is the time of event and d the processu that jumped at time t 
        and m the mark associated to the jump
        Important to note that this algorithm expects the first and last time to mark the beginning and
        the horizon of the observed process. As such, first and last marks must be equal to 0, signifying that they are
        not real event times.
        The algorithm checks by itself if this condition is respected, otherwise it sets the beginning at 0 and the end
        equal to the last time event.
    dim : int
        Number of processes only necessary if providing 1-dimensional theta. Default is None
    dimensional : bool
        Whether to return the sum of loglikelihood or decomposed in each dimension. Default is False.
        
Returns
    -------
    likelihood : array of float
        Value of likelihood at each process.
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """
    
    arg_f = dict(zip(list(name_arg_f) ,x[:len(name_arg_f)]))
    arg_phi = dict(zip(list(name_arg_phi),x[len(name_arg_f) :len(name_arg_f)+len(name_arg_phi)]))
    theta = x[len(name_arg_f)+len(name_arg_phi):]         
    
    
    arg_f = dict(zip(list(name_arg_f) ,x[:len(name_arg_f)]))
    arg_phi = dict(zip(list(name_arg_phi),x[len(name_arg_f) :len(name_arg_f)+len(name_arg_phi)]))
    theta = x[len(name_arg_f)+len(name_arg_phi):]    
    
   
   
   
    if isinstance(theta, np.ndarray):
        if dim is None:
            raise ValueError("Must provide dimension to unpack correctly")
        else:
            mu = np.array(theta[:dim]).reshape((dim, 1))
            a = np.array(theta[dim:dim * (dim + 1)]).reshape((dim, dim))
            b = np.array(theta[dim * (dim + 1):]).reshape((dim, 1))
    else:
        mu, a, b = (i.copy() for i in theta)
    b = b + 1e-10
    b_1 = 1/b

    timestamps = tList.copy()

    # We first check if we have the correct beginning and ending.
    if timestamps[0][1] > 0:
        timestamps = [(0, 0)] + timestamps
    if timestamps[-1][1] > 0:
        timestamps += [(timestamps[-1][0], 0)]

    # Initialise values
    time_b, dim_b, mark_b  = timestamps[1]
    
    # Compensator between beginning and first event time
    compensator = mu*(time_b - timestamps[0][0])
    
    
    # Intensity before first jump
    log_i = np.zeros((a.shape[0],1))
    log_i[dim_b-1] = np.log(mu[dim_b-1])
    
    
    ic = mu + np.multiply(a[:, [dim_b - 1]], phi(mark_b, **arg_phi, **arg_f))
    
    # j=1
    
    for time_c, dim_c, mark_c in timestamps[2:]:
        
        # First we estimate the compensator
        inside_log = (mu - np.minimum(ic, 0))/mu
        
        # Restart time
        t_star = time_b + np.multiply(b_1, np.log(inside_log))

        aux = 1/inside_log  # inside_log can't be equal to zero (coordinate-wise)
        
        #aux = np.minimum(1, aux)
        
        compensator += (t_star < time_c)*(np.multiply(mu, time_c-t_star) + np.multiply(b_1, ic-mu)*(aux - np.exp(-b*(time_c-time_b))))

        # Then, estimation of intensity before next jump.
        if dim_c > 0:
            ic = mu + np.multiply((ic - mu), np.exp(-b*(time_c-time_b))) 

            if ic[dim_c - 1] <= 0.0:

                res = 1e8
                return res
            else:
                log_i[dim_c-1] += np.log(ic[dim_c - 1])

            ic += np.multiply(a[:, [dim_c - 1]], phi(mark_c , **arg_phi,**arg_f))

        time_b = time_c
    likelihood = log_i - compensator

    likelihood += np.sum(list(map(lambda y : np.log(f(y[2], **arg_f)), tList)))

    
    if not(dimensional):
        
        likelihood = np.sum(likelihood)  
        
    return -likelihood
        
    
def loglikelihood(theta, tList, **kwargs):
    """
    Exact computation of the loglikelihood for an exponential Hawkes process for either self-exciting or self-regulating cases. 
    Estimation for a single realization.
    
    Parameters
    ----------
    theta : tuple of float
        Tuple containing the parameters to use for estimation.
    tList : list of float
        List containing all the lists of data (event times).

    Returns
    -------
    likelihood : float
        Value of likelihood, either for 1 realization or for a batch. 
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """
    # Extract variables
    lambda0, a, b = theta
    

    # Avoid wrong values in algorithm such as negative lambda0 or b
    if lambda0 <= 0 or b <= 0:
        return 1e5

    else:

        compensator_k = lambda0 * tList[1]
        lambda_avant = lambda0
        lambda_k = lambda0 + a
       

        if lambda_avant <= 0:
            return 1e5

        likelihood = np.log(lambda_avant) - compensator_k

        # Iteration
        for k in range(2, len(tList)-1):

            if lambda_k >= 0:
                C_k = lambda_k - lambda0
                tau_star = tList[k] - tList[k - 1]
            else:
                C_k = -lambda0
                tau_star = tList[k] - tList[k - 1] - (np.log(-(lambda_k - lambda0)) - np.log(lambda0)) / b

            lambda_avant = lambda0 + (lambda_k - lambda0) * np.exp(-b * (tList[k] - tList[k - 1]))
            lambda_k = lambda_avant + a
            compensator_k = lambda0 * tau_star + (C_k / b) * (1 - np.exp(-b * tau_star))
            
            
            
            if lambda_avant <= 0:
                return 1e5
            
            

            likelihood += np.log(lambda_avant) - compensator_k
        
        k = len(tList)-1
        
        if lambda_k >= 0:
            C_k = lambda_k - lambda0
            tau_star = tList[k] - tList[k - 1]
        else:
            C_k = -lambda0
            tau_star = tList[k] - tList[k - 1] - (np.log(-(lambda_k - lambda0)) - np.log(lambda0)) / b

        compensator_k = lambda0 * tau_star + (C_k / b) * (1 - np.exp(-b * tau_star))
        
        likelihood -=  compensator_k
        # We return the opposite of the likelihood in order to use minimization packages.
        
        return (-likelihood)


def multivariate_loglikelihood(theta, tList, dim=None, dimensional=False):
    """

    Parameters
    ----------
    theta : tuple of array
        Tuple containing 3 arrays. First corresponds to vector of baseline intensities mu. Second is a square matrix
        corresponding to interaction matrix a. Last is vector of recovery rates b.

    tList : list of tuple
        List containing tuples (t, m) where t is the time of event and m is the mark (dimension). The marks must go from
        1 to nb_of_dimensions.
        Important to note that this algorithm expects the first and last time to mark the beginning and
        the horizon of the observed process. As such, first and last marks must be equal to 0, signifying that they are
        not real event times.
        The algorithm checks by itself if this condition is respected, otherwise it sets the beginning at 0 and the end
        equal to the last time event.
    dim : int
        Number of processes only necessary if providing 1-dimensional theta. Default is None
    dimensional : bool
        Whether to return the sum of loglikelihood or decomposed in each dimension. Default is False.
Returns
    -------
    likelihood : array of float
        Value of likelihood at each process.
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """

    if isinstance(theta, np.ndarray):
        if dim is None:
            raise ValueError("Must provide dimension to unpack correctly")
        else:
            mu = np.array(theta[:dim]).reshape((dim, 1))
            a = np.array(theta[dim:dim * (dim + 1)]).reshape((dim, dim))
            b = np.array(theta[dim * (dim + 1):]).reshape((dim, 1))
    else:
        mu, a, b = (i.copy() for i in theta)
    b = b + 1e-10

    b_1 = 1/b

    timestamps = tList.copy()

    # We first check if we have the correct beginning and ending.
    if timestamps[0][1] > 0:
        timestamps = [(0, 0)] + timestamps
    if timestamps[-1][1] > 0:
        timestamps += [(timestamps[-1][0], 0)]

    # Initialise values
    tb, mb = timestamps[1]
    
    # Compensator between beginning and first event time
    compensator = mu*(tb - timestamps[0][0])
    # Intensity before first jump
    log_i = np.zeros((a.shape[0],1))
    log_i[mb-1] = np.log(mu[mb-1])
    ic = mu + a[:, [mb - 1]]
    # j=1

    for tc, mc in timestamps[2:]:


        # First we estimate the compensator
        inside_log = (mu - np.minimum(ic, 0))/mu
        # Restart time
        t_star = tb + np.multiply(b_1, np.log(inside_log))

        aux = 1/inside_log  # inside_log can't be equal to zero (coordinate-wise)
        aux = np.minimum(1, aux)
        
        
        compensator += (t_star < tc)*(np.multiply(mu, tc-t_star) + np.multiply(b_1, ic-mu)*(aux - np.exp(-b*(tc-tb))))

        
        # Then, estimation of intensity before next jump.
        if mc > 0:
            ic = mu + np.multiply((ic - mu), np.exp(-b*(tc-tb)))

            if ic[mc - 1] <= 0.0:
                # print("oh no")
                # res = 1e8

                # rayon_spec = np.max(np.abs(np.linalg.eig(np.abs(a) / b)[0]))
                # rayon_spec = min(rayon_spec, 0.999)
                # if 2*(tList[-1][0])/(1-rayon_spec) < 0:
                #     print("wut")
                # res = 2*(tList[-1][0])/(1-rayon_spec)

                res = 1e8 #*(np.sum(mu**2) + np.sum(a**2) + np.sum(b**2))
                return res
            else:
                log_i[mc-1] += np.log(ic[mc - 1])
                
            # j += 1
            # print(j, ic, 1+0.45*(1-0.5**(j-1)))
            ic += a[:, [mc - 1]]

        tb = tc
    likelihood = log_i - compensator
    if not(dimensional):
        likelihood = np.sum(likelihood)
    return -likelihood

"""
    FIN
"""




