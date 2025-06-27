"""
TEST ON UNIDIM LINEAR HAWKES PROCESS USING HESSIAN OF LOG-LIK
"""
#%%

import pickle as pk
import numpy as np
import functions as hp 
import json
from functions.LogLikHessian import likelihood_hessien_unidim

# Parameters of the Hawkes Process

file_path ="params/params_LEHP.pydata"

with open(file_path, "r") as f:
    params = json.load(f)


alpha = params['alpha']
beta = params['beta']
lambda0 = params['lambda0']
Tmax = params['Tmax']
Nbsample= params['NbSample']


param_estim= []
var_estim = []
TimestampsList = []

NbTrial=0


while ( NbTrial <= Nbsample ):
    

    ## Hawkes Simulation
    hawkes = hp.exp_thinning_hawkes(lambda0, alpha, beta, max_time=Tmax)
    hawkes.simulate()


    
    ## Hawkes Estimation
    
    learner= hp.loglikelihood_estimator(a_bound= None)
    learner.fit(hawkes.timestamps,max_time=True)
    
    
    param_estim+=[learner.theta_estim]
    
    NbTrial+= 1

    
    ## Hessian loglok computation $

    a = likelihood_hessien_unidim(learner.theta_estim[0], 
                           learner.theta_estim[1], 
                           learner.theta_estim[2],
                           np.array(hawkes.timestamps)[1:], 
                           Tmax = Tmax)
    
    

        
          
           
    var_estim += [np.sqrt(np.diag(np.linalg.inv(a)))]
    
        
    param_estim+= [learner.theta_estim]
    
    
    NbTrial+= 1
    
    
results = [param_estim,var_estim]

#with open(f"./simulated_data/Test_Coeff_with_Hessian.pickle", 'wb') as f:
#    pk.dump(results, f)

# %%
