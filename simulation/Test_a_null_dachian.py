
"""

Test Alpha = 0 using the framework where mu and beta are known 

"""

#%%
## module importation 

import numpy as np
import functions as hp
from functions.LogLikHessian import likelihood_hessien_unidim
import csv
import json


## risk type under consideration
risk_type = 1


### Select parameters accordigly

if risk_type == 1:
    file_path ="params/params_poisson.pydata"
else : 
    file_path = "params/params_LEHP.pydata"

## Parameters
with open(file_path, "r") as f:
    params = json.load(f)



alpha = params['alpha'] # a = 0  or 0.6 
beta = params['beta'] # beta = 2
lambda0 = params['lambda0'] # m = 1
Tmax = params['Tmax'] # Tmax = 5000
Nbsample= params['NbSample'] # Nbsample = 5000

param_estim_D, param_estim= [],[]
fisher_info_D, fisher_info= [],[]

def fisher_info_dashan( mu, alpha, beta, tList):

    lambdaTk =[mu]
    partial_lambda_Tk =[0]

    last_time = tList[1]

    for time in tList[2:-1]:

        lambdaTk += [mu + (lambdaTk[-1]+alpha-mu)*np.exp( -beta*(time- last_time))]
        partial_lambda_Tk += [( partial_lambda_Tk[-1] +1)*np.exp( -beta*(time- last_time))]
        last_time = time

    fisher_info = np.sum( np.array(lambdaTk)**(-2) *np.array(partial_lambda_Tk)**2)
    return(fisher_info/Tmax) 

 

for k in range(Nbsample):
    
    if k % 200 == 0 :
        print(k)
        
    hawkes = hp.exp_thinning_hawkes(lambda0, alpha, beta, max_time=Tmax)
    hawkes.simulate()
    
    tList = hawkes.timestamps 

    learner = hp.estimator_unidim_daichan(a_bound = None, mu = lambda0, beta=beta)

    alpha_estim_dachian= learner.fit(tList)
    
    learner = hp.loglikelihood_estimator(a_bound = None)
    alpha_estim = learner.fit(tList, max_time =True)
    
        
    param_estim_D+= [alpha_estim_dachian[0]]
    fisher_info_D += [fisher_info_dashan(lambda0, alpha_estim[0], beta, tList)]

    param_estim+= [alpha_estim[1]]
    fisher_info += [likelihood_hessien_unidim(alpha_estim[0], alpha_estim[1], alpha_estim[2], np.array(tList), Tmax = Tmax)]

        
        

results = np.concatenate( (np.array(param_estim_D).reshape(Nbsample,1), np.array(fisher_info_D).reshape(Nbsample,1)), axis=1)


with open('./simulated_data/Test_alpha_null_unidim_dashan.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(results)
    

# %%
