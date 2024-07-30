"""
    Fonctions computing the values of the compensator for different models 
"""
import numpy as np

def poisson_compensator(tList,theta, **kwargs):
    return([theta[0]*time for time in tList[1:]])


def unidim_MEHP_compensator(tList, theta, phi=lambda mark, t : 1, arg_f={}, arg_phi={}):
    
    
    """
    Compute the compensator in each jump time for a unidim marked exponential hawkes process
    
    Parameters
    ----------
    theta : tuple of float
        Tuple containing the parameters of the hawkes.
        
    tList : list of tuple
        List of tuple containing all the event times and the value of the mark at each timestamps.
        The first tuple must contains the time t0 at which the simulation begins and the last tuple must contains the value of Tmax, maximum time for which the process has been record. 
        
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
    a = theta[1]
    b = theta[2]
        


    b_1 = 1/b

    transformed_times = []

    # Initialise values
    last_time = tList[1][0]
    
    # Compensator between beginning and first event time
    
    compensator = mu*(last_time - tList[0][0])
    transformed_times += [compensator]
    
    
    # Intensity 
    ic = mu + a*phi( tList[0][1], **arg_phi, **arg_f)
    
    

    for time, mark in tList[2:]:
                
        # First we estimate the compensator
        
        inside_log = (mu - np.minimum(ic, 0))/mu
        
        # Restart time
        t_star = last_time +b_1*np.log(inside_log)
        
        aux = 1/inside_log 
        
        
        compensator = (t_star < time)*(mu*(time-t_star) + b_1*(ic-mu)*(aux - np.exp(-b*(time-last_time))))
        transformed_times += [transformed_times[-1]+compensator]

        
        ic = mu + (ic  - mu)*np.exp(-b*(time-last_time)) + a*phi(mark, **arg_phi, **arg_f)

        last_time= time
                
    return transformed_times

def unidim_EHP_compensator(tList,theta, **kwargs):

    
    """
    Compute the compensator in each jump time for a unidim exponential hawkes process
    
    Parameters
    ----------
    theta : tuple of float
        Tuple containing the parameters of the hawkes.
        
    tList : list of float
        List containing all the event times. 
        The first time must contains the time t0 at which the recording of the process begins and the last time must be the value of Tmax, maximum time for which the process has been record. 

    Returns
    -------
        List of float 
    """

    mu = theta[0]
    a = theta[1]
    b = theta[2]   


    b_1 = 1/b

    transformed_times = []

    # Initialise values
    tb = tList[1]
    
    # Compensator between beginning and first event time
    
    compensator = mu*(tb - tList[0])
    transformed_times += [compensator]
    
    
    # Intensity 
    ic = mu + a
    

    for tc in tList[2:]:
        
        # First we estimate the compensator
        
        inside_log = (mu - np.minimum(ic, 0))/mu
        
        # Restart time
        t_star = tb +b_1*np.log(inside_log)
        
        aux = 1/inside_log 
        
        
        compensator = (t_star < tc)*(mu*(tc-t_star) + b_1*(ic-mu)*(aux - np.exp(-b*(tc-tb))))
        transformed_times+= [transformed_times[-1] + compensator]
        
        
        ic = mu + (ic - mu)*np.exp(-b*(tc-tb))  + a

        tb = tc
        
        
    return np.array(transformed_times)


def multi_EHP_compensator(tList, theta,**kwargs):


    """
    Compute the compensator in each jump time for a multidim exponential hawkes process
    
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
    
    if isinstance(theta, np.ndarray):
        
        dim = int(np.sqrt(1 + theta.shape[0]) - 1)
        
        mu = np.array(theta[:dim]).reshape((dim, 1))
        a = np.array(theta[dim:dim * (dim + 1)]).reshape((dim, dim))
        b = np.array(theta[dim * (dim + 1):]).reshape((dim, 1))
        
    else:
        mu, a, b = (i.copy() for i in theta)
        dim = len(mu)
          

    b_1 = 1/b

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
    ic = mu + a[:, [mb - 1]]
    # j=1

    for tc, mc in tList[2:]:
        # First we estimate the compensator
        inside_log = (mu - np.minimum(ic, 0))/mu
        # Restart time
        t_star = tb + np.multiply(b_1, np.log(inside_log))

        aux = 1/inside_log  # inside_log can't be equal to zero (coordinate-wise)
        #aux = np.minimum(1, aux)
        compensator = (t_star < tc)*(np.multiply(mu, tc-t_star) + np.multiply(b_1, ic-mu)*(aux - np.exp(-b*(tc-tb))))

        transformed_times += [np.sum(compensator)]
        counter += compensator
        individual_transformed_times[mc - 1] += [counter[mc - 1, 0]]
        counter[mc - 1] = 0

        # Then, estimation of intensity before next jump.
        if mc > 0:
            ic = mu + np.multiply((ic - mu), np.exp(-b*(tc-tb)))
            ic += a[:, [mc - 1]]

        tb = tc
    #print("transformed_times", individual_transformed_times[1][0:10])
    return transformed_times, individual_transformed_times


def multi_MEHP_compensator(tList,theta, phi={}, arg_phi={}, arg_f={}):
    
    if isinstance(theta, np.ndarray):
                
        dim = int(np.sqrt(1 + theta.shape[0]) - 1)
        
        mu = np.array(theta[:dim]).reshape((dim, 1))
        a = np.array(theta[dim:dim * (dim + 1)]).reshape((dim, dim))
        b = np.array(theta[dim * (dim + 1):]).reshape((dim, 1))
        
    else:
        mu, a, b = (i.copy() for i in theta)
        dim = len(mu)
          

    b_1 = 1/b

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
    ic = mu + a[:, [dim_b - 1]]*(phi(mark_b, **arg_phi, **arg_f)[:, [dim_b - 1]])
    # j=1

    for time_c, dim_c, mark_c in tList[2:]:
        # First we estimate the compensator
        inside_log = (mu - np.minimum(ic, 0))/mu
        # Restart time
        t_star = time_b + np.multiply(b_1, np.log(inside_log))

        aux = 1/inside_log  # inside_log can't be equal to zero (coordinate-wise)
        #aux = np.minimum(1, aux)
        compensator = (t_star < time_c)*(np.multiply(mu, time_c-t_star) + np.multiply(b_1, ic-mu)*(aux - np.exp(-b*(time_c-time_b))))

        transformed_times += [np.sum(compensator)]  

        counter += compensator
        individual_transformed_times[dim_c - 1] += [counter[dim_c - 1, 0]]
        counter[dim_c - 1] = 0

        # Then, estimation of intensity before next jump.
        if dim_c > 0:
            ic = mu + np.multiply((ic - mu), np.exp(-b*(time_c-time_b)))
            ic += a[:, [dim_c - 1]]*(phi(mark_c, **arg_phi,**arg_f)[:, [dim_c - 1]])
            

        time_b = time_c
    #print("transformed_times", individual_transformed_times[1][0:10])
    return transformed_times, individual_transformed_times
