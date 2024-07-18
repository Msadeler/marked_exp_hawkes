"""
    Functions to perform the GOF procedure with bootstrap
"""

import numpy as np 
from scipy.stats import kstest


def aggregated_process(timeslists):
    aggList = [0.0]
    for times in timeslists:
        aggList += [aggList[-1] + i for i in times]

    max_time = np.cumsum([len(times) for times in timeslists])
    return aggList, max_time


def GOF_procedure(index_sample,theta, tList, markList, compensator_func,sup_compensator, phi = lambda x,a : 1, arg_f = {}, arg_phi={}):


    ### computation of the cumulated process

    time_transformed = [ compensator_func(theta = theta,tList = tList[subset], markList= markList[subset], phi = phi, arg_phi = arg_phi, arg_f = arg_f) for subset in index_sample ]

    #%%
    cumulated_process, max_time  = aggregated_process(time_transformed)
    starting_time = cumulated_process[-1]
    process_unif = np.array([cumulated_process[k] for k in np.arange(max_time[-1]) if k not in max_time])

    if sup_compensator>starting_time:
        print("The chosen born is greater than the actual founded : sup_taken = {} and sup founded = {}".format(sup_compensator,starting_time))


    selected_time = process_unif[process_unif<= sup_compensator]/sup_compensator
    
    
    ### Test if pval follow an uniform law on [0,1]
    pval = kstest(selected_time, cdf='uniform').pvalue


    return(pval)

# %%
