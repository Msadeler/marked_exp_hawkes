
#%%
from functions.hawkes_process import *
from functions.GOF import *
from functions.compensator import *
from functions.estimator_class import loglikelihood_estimator_bfgs
from functions.estimator_class_multi_rep import *

np.random.seed(0)


#%%

if __name__ == '__main__': 
    
    print("MODULE IMPORTATION OK")
    tlist = []

    for k in range(50):
        process = exp_thinning_hawkes(1, 0.6, 1 , max_time=50)
        process.simulate()
        timestamps = process.timestamps 

        tlist+= [timestamps]


    print('END SIMU')


    learner = estimator_unidim_multi_rep()
    res = learner.fit(tlist,nb_cores = 3)

    print('END')
    
    
    stats = learner.GOF_bootstrap(nb_cores = 3)
# %%
