"""
    Functions to perform the GOF procedure with bootstrap
"""

import numpy as np 
from scipy.stats import kstest


def aggregated_process(timeslists: list):
    
    ### aggregate all process given
    aggList = [0.0]
    for times in timeslists:
        aggList += [aggList[-1] + i for i in times]

    ## delete the time corresponding to Lambd(Tmax) that are not actual jump time
    end_times = np.cumsum([len(times) for times in timeslists])
    max_obs_time = aggList[-1]
    
    aggList = np.array([aggList[k] for k in np.arange(end_times[-1]) if k not in end_times])
    
    return aggList,max_obs_time


def GOF_procedure(time_transformed_list: list, sup_compensator: float, test_type : str):


    ### computation of the cumulated process
    cumulated_process,end_time  = aggregated_process(time_transformed_list)

    if not sup_compensator: 
        sup_compensator = 0.9*end_time
        
    elif sup_compensator>end_time:
        print("The chosen born is greater than the actual founded : sup_taken = {} and sup founded = {}".format(sup_compensator,end_time))

    if test_type =='uniform':
        selected_time = cumulated_process[cumulated_process<= sup_compensator]/sup_compensator
    elif test_type =='expon':
        selected_time = cumulated_process[2:]- cumulated_process[1:-1]
    
    
    ### Test if pval follow an uniform law on [0,1]
    pval = kstest(selected_time, cdf=test_type).pvalue


    return(pval)

# %%
