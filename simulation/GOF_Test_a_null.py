#%%
import numpy as np
import functions as hp
import matplotlib.pyplot as plt
import json

file_path = "params/params_LEHP.pydata"

## Parameters
with open(file_path, "r") as f:
    params = json.load(f)

output = []
outputHawkes=[]

m,b = params['lambda0'], params['beta']
Tmax = params['Tmax']
Nbsample= params['NbSample']

for a in  [0.05,0.1,0.4]:


    process = hp.exp_thinning_hawkes_multi_marked(m=m,
                                            a=a, 
                                            b=b, 
                                            max_time=Tmax,
                                            n=Nbsample)
    process.simulate()


    
    learner = hp.estimator_unidim_multi_rep(loss = hp.likelihood_Poisson )
    learner.fit(process.timeList)

    
    test_poisson = learner.GOF_bootstrap(compensator_func=hp.poisson_compensator,
                                         Nb_SubSample=int(Nbsample**(2/3)))
    

    learner_hawkes = hp.estimator_unidim_multi_rep()
    learner_hawkes.fit(process.timeList)
    test_hawkes = learner_hawkes.GOF_bootstrap(Nb_SubSample=int(Nbsample**(2/3)))


    output +=[test_poisson['pvalList']]
    outputHawkes +=[test_hawkes['pvalList']]


# %%
