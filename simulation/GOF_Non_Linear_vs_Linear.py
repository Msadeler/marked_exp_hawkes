
#################################################################################################################
#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import functions as hp
import json
#%%

file_path ="params/params_LEHP.pydata"

with open(file_path, "r") as f:
    params = json.load(f)


alpha = params['alpha']
beta = params['beta']
lambda0 = params['lambda0']
Tmax = params['Tmax']
Nbsample= params['NbSample']


SubSample = int(Nbsample**(2/3) )



tList = []

Nbsample = 2

for k in range(Nbsample):
    hawkes_ex =   hp.exp_thinning_hawkes(m=lambda0, a = alpha, b=beta, max_time=Tmax)
    hawkes_ex.simulate()
    tList += [hawkes_ex.timestamps]

#%%

learner_risk_type1 = hp.estimator_unidim_multi_rep(a_bound=None)
learner_risk_type2 = hp.estimator_unidim_multi_rep(a_bound=0)

learner_risk_type1.fit(tList)
learner_risk_type2.fit(tList)


#%%
pval_risktype1 = learner_risk_type1.GOF_procedure(sup_compensator=100, SubSample_size=SubSample, Nb_SubSample=Nbsample)
pval_risktype2 = learner_risk_type2.GOF_procedure(sup_compensator=100, SubSample_size=SubSample, Nb_SubSample=Nbsample)




#with open('./simulated_data/GOF_procedure_Linear_vs_NLinear.csv', 'w', newline='') as file:  
#    writer = csv.writer(file)
#    writer.writerows(np.array([pval_risktype1['pvalList'],pval_risktype2['pvalList']]).T)
    
