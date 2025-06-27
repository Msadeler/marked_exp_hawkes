##########################################################################################################
########################  COMPARISON OF DIFFERENTS PROCEDURE OF TEST WITH MARK  ##########################
##########################################################################################################

#%%

### Modules
import json
import numpy as np
from scipy.stats import kstest
import pandas as pd
from functions.estimator_class import *
import scipy
import functions as hp


file_path = "params/params_LMEHP.pydata"

## Parameters
with open(file_path, "r") as f:
    params = json.load(f)

output = []
outputHawkes=[]

m,a,b = params['lambda0'],params['alpha'], params['beta']
gamma= params['gamma']
psi = params['psi']

Tmax = params['Tmax']
Nbsample= params['NbSample']


#%%

SubSample = int(Nbsample**(2/3) )


from functions.paramtrised_function import *




hawkes = hp.exp_thinning_hawkes_multi_marked(m=m,
                                        a=a, 
                                        b=b, 
                                        mark_process=True,
                                        F=F, 
                                        arg_F={'psi':psi}, 
                                        phi=phi, 
                                        arg_phi={'gamma':gamma}, 
                                        max_jumps=Tmax, 
                                        n=Nbsample)
hawkes.simulate()

#%%
learnerMHPCo = hp.estimator_unidim_multi_rep(mark=True, 
                                     f=f, 
                                    name_arg_f=['psi'], 
                                    phi=phi, 
                                    name_arg_phi=['gamma'], 
                                    bound_f=[(1e-5, None)], 
                                    bound_phi=[(None,None)], 
                                    initial_guess_f=[1], 
                                    initial_guess_phi=[0])

learnerMHPCo.fit(hawkes.timeList)
statCorrect = learnerMHPCo.GOF_bootstrap(compensator_func=unidim_MEHP_compensator,
                                            test_type = 'uniform', 
                                              Nb_SubSample=SubSample, 
                                              plot = True)
#%%
learnerIC = hp.estimator_unidim_multi_rep(mark=True, 
                                     f=f, 
                                    name_arg_f=['psi'], 
                                    phi=phi1, 
                                    name_arg_phi=['gamma'], 
                                    bound_f=[(1e-5, None)], 
                                    bound_phi=[(None,None)], 
                                    initial_guess_f=[1], 
                                    initial_guess_phi=[0])
learnerIC.fit(hawkes.timeList)

#%%
statInCorrect = learnerIC.GOF_bootstrap(compensator_func=unidim_MEHP_compensator,
                                        test_type = 'uniform', 
                                              Nb_SubSample=SubSample, 
                                              plot = True)

#%%

tlist = [[time for time,mark in timelist] for timelist in hawkes.timeList]

#%%
learnerHP= hp.estimator_unidim_multi_rep()
learnerHP.fit(tlist)
statHP= learnerHP.GOF_bootstrap(compensator_func=unidim_EHP_compensator,
                                test_type = 'uniform', 
                                              Nb_SubSample=SubSample, 
                                              plot = True)


