#%%


import functions as hp
import numpy as np
from functions.paramtrised_function import *
import time



file_path = "params/params_LMEHP.pydata"

## Parameters
with open(file_path, "r") as f:
    params = json.load(f)

output = []
outputHawkes=[]

m,a,b = params['lambda0'],params['alpha'], params['beta'] ## m = 1, a  0.6, b = 2
gamma= params['gamma'] # gamma = 2
psi = params['psi'] # psi = 1
n_rep= params['NbSample'] ## nbsample = 150

tmax =[50, 100, 200, 500, 1000]



#%%
list_time = []

for time_max in tmax:
    time_simu = 0 
    time_test = 0 

    for k in range(20):
        
        hawkes = hp.exp_thinning_hawkes_multi_marked(m,a,b, 
                                                    mark_process=True,
                                                    phi = phi,
                                                    F = F, 
                                                    arg_phi = {'gamma':gamma},
                                                    arg_F={'psi':psi}, 
                                                    max_time=time_max,
                                                    n=n_rep)
        hawkes.simulate()
        print(time_max)

        tlistnomark = [[time for time, mark in tlist] for tlist in hawkes.timeList]
        learner_MEHP = hp.estimator_unidim_multi_rep(loss=hp.likelihood_Poisson)
        start = time.time()   
        learner_MEHP.fit(tlistnomark, max_time=True,nb_cores=1)
        end = time.time() - start

        start_estim = time.time()
        stats_MEHP = learner_MEHP.GOF_bootstrap(compensator_func=hp.poisson_compensator,
                                                test_type = 'uniform', 
                                                Nb_SubSample=int(n_rep**(2/3)),
                                                plot = False,
                                                nb_cores=1)
        end_test = time.time()-start_estim

        time_simu+= end
        time_test += end_test

    list_time+=[[time_simu/20,time_test/20]]

data = {'param_init':[psi,gamma, m,a,b],
        'param_name' : ['psi', 'gamma', 'm', 'a', 'b'],
        'model': 'PP',
        'max_time' : tmax,
        'n_rep': n_rep,
            'phi' : 'exp',
            'f' : 'exp',
        'computation_time': list_time}

#import json
#with open(f'simulated_data/GOF/PP_gamma{gamma}_a{a}.json', 'w') as f:
#    json.dump(data, f)

# %%

import json

paramboot =[]
param =[]
Testa,Testmark = [],[]

def rec_formula(t, m,a,b,arg_f,arg_phi, phi, timebefore, intensity_last_jump, time_transformed):
    return( time_transformed+ m*(t-timebefore[0])+ (1-np.exp(-b*(t-timebefore[0])))*(intensity_last_jump+a*phi(timebefore[1], **arg_phi, **arg_f)-m)/b )


time_list = []

for time_max in tmax:
  
  time_computation = 0

  
  for k in range(20):
      
    hawkes = hp.exp_thinning_hawkes_marked(m,a,b, mark_process=True,
                                            phi = phi, 
                                            F = F, 
                                            arg_phi = {'gamma':gamma}, 
                                            arg_F={'psi':psi}, 
                                            max_time=time_max)
    hawkes.simulate()

    estimator = hp.estimator_bootstrap(mark=True,
                                    name_arg_f=['psi'], 
                                        name_arg_phi=['gamma'], 
                                        f=f,
                                        F=F,
                                        phi=phi,
                                        initial_guess_f = [1], 
                                        initial_guess_phi = [1], 
                                        bound_phi = [(1e-5, None)],
                                        bound_f = [(1e-5,None)])
    strat = time.time()
    estimator.boostrap_procedure( tlist =hawkes.timestamps,B=n_rep,max_time=True,rec_formula=rec_formula, nb_cores=1)
    end = time.time()- strat
    time_computation+= end


  time_list+=[time_computation/20]
  
  with open(f'simulated_data/bootstrap.json', 'w') as f:
    json.dump({'tmax':tmax,
    'time':time_list}, f)
# %%
