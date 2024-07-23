"""
    Functions to perform the GOF procedure with bootstrap
"""

import numpy as np 
from scipy.stats import kstest


def aggregated_process(timeslists):
    aggList = [0.0]
    for times in timeslists:
        aggList += [aggList[-1] + i for i in times]

    end_times = np.cumsum([len(times) for times in timeslists])
    max_obs_time = end_times[-1]
    
    aggList = np.array([aggList[k] for k in np.arange(end_times[-1]) if k not in end_times])
    
    return aggList,max_obs_time


def GOF_procedure(time_transformed_list, sup_compensator):


    ### computation of the cumulated process
    cumulated_process,end_time  = aggregated_process(time_transformed_list)

    if not sup_compensator: 
        sup_compensator = 0.9*end_time
        
    elif sup_compensator>end_time:
        print("The chosen born is greater than the actual founded : sup_taken = {} and sup founded = {}".format(sup_compensator,end_time))


    selected_time = cumulated_process[cumulated_process<= sup_compensator]/sup_compensator
    
    
    ### Test if pval follow an uniform law on [0,1]
    pval = kstest(selected_time, cdf='uniform').pvalue


    return(pval)

# %%
