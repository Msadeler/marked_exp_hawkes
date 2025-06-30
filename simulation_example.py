#%%
from functions.hawkes_process import *
from functions.GOF import *
from functions.compensator import *
from functions.multivariate_exponential_process import *
from functions.estimator_class_multi_rep import *
import json

#####################################################################################################
######################## Exemple of scritp for the simulation and estimation ########################
#####################################################################################################

np.random.seed(0)


file_path = "simulation/params/params_NLEHP.pydata"

## Parameters
with open(file_path, "r") as f:
    params = json.load(f)


m,a,b = params['lambda0'],params['alpha'], params['beta'] ## m = , a = -0.2, b = 1
Tmax = params['Tmax'] ## Tmax = 5000
Nbsample= params['NbSample'] ## Nbsample = 500



## simulation 
hawkes=  exp_thinning_hawkes_multi_marked(m=m,
                                    a=a, 
                                    b=b, 
                                    n=Nbsample,
                                    max_jumps=Tmax)
hawkes.simulate()


learner_hawkes = estimator_unidim_multi_rep(a_bound = None,
                                            bound_b = None)
learner_hawkes.fit(hawkes.timeList, max_jump = True)



stat = learner_hawkes.test_one_coeff( coefficient_index=0, value = 1,plot=True)

# %%
