#%%
##########################################################################################################
############################ Loi asymptotitque des pramètres avec inhibition #############################
##########################################################################################################

import numpy as np
import functions as hp
from functions.LogLikHessian import likelihood_hessien_unidim
import multiprocessing
import time
import os
import json

file_path ="params/params_inhib.pydata"

with open(file_path, "r") as f:
    params = json.load(f)


#%%
alpha = params['beta']
beta = params['alpha']
lambda0 = params['lambda0']
gamma= params['gamma']
Tmax = params['Tmax']
NbSample= params['NbSample'] 

AlphaList = []
Hessian = []

#%%

def param_law_inhib(k):
    
    np.random.seed((os.getpid() * int(time.time())) % 123456789)    

    
    hawkes = hp.exp_thinning_hawkes(lambda0, alpha, beta, max_time=Tmax)
    hawkes.simulate()
    
    tList = hawkes.timestamps + [Tmax]
    
    
    learner_no_mark = hp.loglikelihood_estimator_bfgs(alpha_bound = None )
        
    param_no_mark = learner_no_mark.fit(tList)
    hessian = likelihood_hessien_unidim(param_no_mark[0], param_no_mark[1], param_no_mark[2], np.array(tList), Tmax = Tmax)
    return(param_no_mark,hessian)
    


pool = multiprocessing.Pool(16)                         
param_inhib = pool.map(param_law_inhib, [k for k in range(1000)])

param = [x[0] for x in param_inhib]


#with open('./simulated_data/AsymptoticLawParamInhib.csv', 'w', newline='') as file:
#    writer = csv.writer(file)
#    writer.writerows(np.array(param))
 

# %%
