##############################################################################
##############################################################################
######################## Test coeff beta_i = beta_j ##########################
##############################################################################
##############################################################################

#%%%

import functions as hp
import pickle as pk
import numpy as np 
import json 

with open('params/params_MultidimLEHP.pydata', "r") as f:
    params = json.load(f)

#%%


risk_type = 2

alpha = params['alpha'] ## a = [[0.4, 0.2],[0.2, 0.3]]
beta = beta = np.array(params["beta"][str(risk_type)]) ## b = [[1],[1]] or [[1],[1.5]]
lambda0 = params['lambda0'] # m = [[1], [1]]
Tmax = params['Tmax'] # Tmax = 5000
Nbsample= params['NbSample'] # nbsample = 5000
dim = params['dim'] ## dim = 2

NbTrial = 0

param_estim = []

while ( NbTrial < Nbsample ):
    
    hawkes = hp.multivariate_exponential_hawkes(lambda0, alpha, beta, max_time=Tmax)
    hawkes.simulate()
    


    tList = hawkes.timestamps
        
    learner = hp.multivariate_estimator_bfgs(dimension=dim, 
                                          alpha_bound = None)
    
    learner.fit(tList)
    
        
    param_estim+= [learner.theta_estim]
            
    NbTrial+=1


#with open(f"./simulated_data/Estimator_Test_Equality_Beta_RiskType_{int(risk_type)}.csv", 'wb') as f:
#    pk.dump(param_estim, f)
    