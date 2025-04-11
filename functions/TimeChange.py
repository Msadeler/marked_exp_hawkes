import numpy as np 
import scipy.integrate as integrate
from scipy.stats import kstest
import os
import time



def GOF_bootstrap(index_sample,theta, tList, markList, compensator_func,sup_compensator, phi = lambda x : 1, arg_f = {}, arg_phi={}):


    ### computation of the cumulated process
    time_transfored_i = compensator_func(theta,tList[index_sample[0]], markList[index_sample[0]], phi=phi, arg_f=arg_f, arg_phi=arg_phi)
    time_transformed_cumulated =  time_transfored_i[:-1]
    starting_time = time_transfored_i[-1]

    for subset in index_sample[1:]:
        time_transfored_i =  compensator_func(theta,tList[subset], markList[subset], phi, arg_f, arg_phi) + starting_time
        time_transformed_cumulated = np.concatenate((time_transformed_cumulated, time_transfored_i[:-1]))
        starting_time = time_transfored_i[-1]

    
    if sup_compensator>starting_time:
        print("The chosen born is greater than the actual founded : sup_taken = {} and sup founded = {}".format(sup_compensator,starting_time))


    selected_time = time_transformed_cumulated[time_transformed_cumulated<= sup_compensator]/sup_compensator
    
    
    ### Test if pval follow an uniform law on [0,1]
    pval = kstest(selected_time, cdf='uniform').pvalue


    return(pval)


def time_change_poisson(theta, tList, markList, phi, arg_f, arg_phi):
    return([theta[0]*time for time in tList[1:]])

def time_change_mark_unidim_diff(theta, tList, markeList, phi=lambda mark, t : 1, arg_f={}, arg_phi={}):
    
    
    """
    Compute the compensator in each jump time for a unidim marked hawkes process
    
    Parameters
    ----------
    theta : tuple of float
        Tuple containing the parameters of the hawkes.
        
    tList : list of float
        List containing all the event times.
        
    markeList: list of float 
        List containing the value of the mark at the time of each jump in tList
        
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
        List of float 
    """
   
                
    mu = theta[0]
    alpha = theta[1]
    beta = theta[2]
        


    beta_1 = 1/beta

    transformed_times = []

    # Initialise values
    last_time = tList[1]
    
    # Compensator between beginning and first event time
    
    compensator = mu*(last_time - tList[0])
    transformed_times += [compensator]
    
    
    # Intensity 
    ic = mu + alpha*phi(markeList[1], **arg_phi, **arg_f)
    
    

    for time, mark in zip(tList[2:-1], markeList[1:-1]):
                
        # First we estimate the compensator
        
        inside_log = (mu - np.minimum(ic, 0))/mu
        
        # Restart time
        t_star = last_time +beta_1*np.log(inside_log)
        
        aux = 1/inside_log 
        
        
        compensator = (t_star < time)*(mu*(time-t_star) + beta_1*(ic-mu)*(aux - np.exp(-beta*(time-last_time))))
        transformed_times += [compensator]

        
        ic = mu + (ic  - mu)*np.exp(-beta*(time-last_time)) + alpha*phi(mark, **arg_phi, **arg_f)

        last_time= time
                
    return transformed_times


def time_change_mark_unidim(theta, tList, markeList, phi=lambda mark, t : 1, arg_f={}, arg_phi={}):
    
    
    """
    Compute the compensator in each jump time for a unidim marked hawkes process
    
    Parameters
    ----------
    theta : tuple of float
        Tuple containing the parameters of the hawkes.
        
    tList : list of float
        List containing all the event times.
        
    markeList: list of float 
        List containing the value of the mark at the time of each jump in tList
        
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
        List of float 
    """
   
                
    mu = theta[0]
    alpha = theta[1]
    beta = theta[2]
        


    beta_1 = 1/beta

    transformed_times = []

    # Initialise values
    last_time = tList[1]
    
    # Compensator between beginning and first event time
    
    compensator = mu*(last_time - tList[0])
    transformed_times += [compensator]
    
    
    # Intensity 
    ic = mu + alpha*phi(markeList[1], **arg_phi, **arg_f)
    
    

    for time, mark in zip(tList[2:-1], markeList[2:-1]):
                
        # First we estimate the compensator
        
        inside_log = (mu - np.minimum(ic, 0))/mu
        
        # Restart time
        t_star = last_time +beta_1*np.log(inside_log)
        
        aux = 1/inside_log 
        
        
        compensator = (t_star < time)*(mu*(time-t_star) + beta_1*(ic-mu)*(aux - np.exp(-beta*(time-last_time))))
        transformed_times += [transformed_times[-1]+compensator]

        
        ic = mu + (ic  - mu)*np.exp(-beta*(time-last_time)) + alpha*phi(mark, **arg_phi, **arg_f)

        last_time= time
                
    return transformed_times

def time_change_unidim_unmarked(theta, tList):


    mu = theta[0]
    alpha = theta[1]
    beta = theta[2]
        


    beta_1 = 1/beta

    transformed_times = []

    # Initialise values
    tb = tList[1]
    
    # Compensator between beginning and first event time
    
    compensator = mu*(tb - tList[0])
    transformed_times += [compensator]
    
    
    # Intensity 
    ic = mu + alpha
    

    for tc in tList[2:]:
        
        # First we estimate the compensator
        
        inside_log = (mu - np.minimum(ic, 0))/mu
        
        # Restart time
        t_star = tb +beta_1*np.log(inside_log)
        
        aux = 1/inside_log 
        
        
        compensator = (t_star < tc)*(mu*(tc-t_star) + beta_1*(ic-mu)*(aux - np.exp(-beta*(tc-tb))))
        transformed_times+= [transformed_times[-1] + compensator]
        
        
        ic = mu + (ic - mu)*np.exp(-beta*(tc-tb))  + alpha

        tb = tc
        
        
    return np.array(transformed_times)




  
def time_change_unidim_diff(theta, tList):
    
    
    """
    Compute the compensator in each jump time for a unidim hawkes process
    
    Parameters
    ----------
    theta : tuple of float
        Tuple containing the parameters of the hawkes.
        
    tList : list of float
        List containing all the event times.
        
    Returns
    -------
        List of float 
    """
   
                
    mu = theta[0]
    alpha = theta[1]
    beta = theta[2]
        


    beta_1 = 1/beta

    transformed_times = []

    # Initialise values
    tb = tList[1]
    
    # Compensator between beginning and first event time
    
    compensator = mu*(tb - tList[0])
    transformed_times += [compensator]
    
    
    # Intensity 
    ic = mu + alpha
    
    

    for tc in tList[2:]:
        
        # First we estimate the compensator
        
        inside_log = (mu - np.minimum(ic, 0))/mu
        
        # Restart time
        t_star = tb +beta_1*np.log(inside_log)
        
        aux = 1/inside_log 
        
        
        compensator = (t_star < tc)*(mu*(tc-t_star) + beta_1*(ic-mu)*(aux - np.exp(-beta*(tc-tb))))
        transformed_times+= [compensator]
        
        
        ic = mu + (ic - mu)*np.exp(-beta*(tc-tb))  + alpha

        tb = tc
        
        
    return np.array(transformed_times)


def time_change_multidim(theta, tList):
    
    if isinstance(theta, np.ndarray):
        
        dim = int(np.sqrt(1 + theta.shape[0]) - 1)
        
        mu = np.array(theta[:dim]).reshape((dim, 1))
        alpha = np.array(theta[dim:dim * (dim + 1)]).reshape((dim, dim))
        beta = np.array(theta[dim * (dim + 1):]).reshape((dim, 1))
        
    else:
        mu, alpha, beta = (i.copy() for i in theta)
        dim = len(mu)
          

    beta_1 = 1/beta

    counter = np.zeros((dim, 1))
    transformed_times = []
    individual_transformed_times = [[] for i in range(dim)]

    # Initialise values
    tb, mb = tList[1]
    
    # Compensator between beginning and first event time
    compensator = mu*(tb - tList[0][0])
    transformed_times += [np.sum(compensator)]
    individual_transformed_times[mb-1] += [compensator[mb - 1, 0]]
    # Intensity before first jump
    ic = mu + alpha[:, [mb - 1]]
    # j=1

    for tc, mc in tList[2:]:
        # First we estimate the compensator
        inside_log = (mu - np.minimum(ic, 0))/mu
        # Restart time
        t_star = tb + np.multiply(beta_1, np.log(inside_log))

        aux = 1/inside_log  # inside_log can't be equal to zero (coordinate-wise)
        #aux = np.minimum(1, aux)
        compensator = (t_star < tc)*(np.multiply(mu, tc-t_star) + np.multiply(beta_1, ic-mu)*(aux - np.exp(-beta*(tc-tb))))

        transformed_times += [np.sum(compensator)]
        counter += compensator
        individual_transformed_times[mc - 1] += [counter[mc - 1, 0]]
        counter[mc - 1] = 0

        # Then, estimation of intensity before next jump.
        if mc > 0:
            ic = mu + np.multiply((ic - mu), np.exp(-beta*(tc-tb)))
            ic += alpha[:, [mc - 1]]

        tb = tc
    #print("transformed_times", individual_transformed_times[1][0:10])
    return transformed_times, individual_transformed_times

def time_change_mark_multidim(theta, tList, phi={}, arg_phi={}, arg_f={}):
    
    if isinstance(theta, np.ndarray):
                
        dim = int(np.sqrt(1 + theta.shape[0]) - 1)
        
        mu = np.array(theta[:dim]).reshape((dim, 1))
        alpha = np.array(theta[dim:dim * (dim + 1)]).reshape((dim, dim))
        beta = np.array(theta[dim * (dim + 1):]).reshape((dim, 1))
        
    else:
        mu, alpha, beta = (i.copy() for i in theta)
        dim = len(mu)
          

    beta_1 = 1/beta

    counter = np.zeros((dim, 1))
    transformed_times = []
    individual_transformed_times = [[] for i in range(dim)]

    # Initialise values
    time_b, dim_b, mark_b = tList[1]
    
    # Compensator between beginning and first event time
    compensator = mu*(time_b - tList[0][0])
    transformed_times += [np.sum(compensator)]
    individual_transformed_times[dim_b-1] += [compensator[dim_b - 1, 0]]
    # Intensity before first jump
    ic = mu + alpha[:, [dim_b - 1]]*(phi(mark_b, **arg_phi, **arg_f)[:, [dim_b - 1]])
    # j=1

    for time_c, dim_c, mark_c in tList[2:-1]:
        # First we estimate the compensator
        inside_log = (mu - np.minimum(ic, 0))/mu
        # Restart time
        t_star = time_b + np.multiply(beta_1, np.log(inside_log))

        aux = 1/inside_log  # inside_log can't be equal to zero (coordinate-wise)
        #aux = np.minimum(1, aux)
        compensator = (t_star < time_c)*(np.multiply(mu, time_c-t_star) + np.multiply(beta_1, ic-mu)*(aux - np.exp(-beta*(time_c-time_b))))

        transformed_times += [np.sum(compensator)]  

        counter += compensator
        individual_transformed_times[dim_c - 1] += [counter[dim_c - 1, 0]]
        counter[dim_c - 1] = 0

        # Then, estimation of intensity before next jump.
        if dim_c > 0:
            ic = mu + np.multiply((ic - mu), np.exp(-beta*(time_c-time_b)))
            ic += alpha[:, [dim_c - 1]]*(phi(mark_c, **arg_phi,**arg_f)[:, [dim_c - 1]])
            

        time_b = time_c
    #print("transformed_times", individual_transformed_times[1][0:10])
    return transformed_times, individual_transformed_times

def mark_change_mark_multidim(tList, f, arg_f,borne_inf=-np.inf):
    
    marked_transformed = []
    
    for time, dim, mark  in tList[1:]:
        
        marked_transformed += [integrate.quad(lambda x: f(x,time, **arg_f), borne_inf,mark) ]
        
    return(marked_transformed)
        
  
def mark_change_mark_unidim(tList, markeList, f, arg_f,borne_inf=-np.inf):
    
    
    marked_transformed = []
    
    for i in range(1,len(tList)-1):
        
        marked_transformed += [integrate.quad(lambda x: f(x,tList[i], **arg_f), borne_inf, markeList[i]) ]
        
    return(marked_transformed)
        
  
