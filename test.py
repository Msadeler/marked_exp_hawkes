
#%%
from functions.hawkes_process import *
from functions.GOF import *
from functions.compensator import *
from functions.estimator_class import loglikelihood_estimator_bfgs
from functions.estimator_class_multi_rep import *


np.random.seed(0)


tlist = []

for k in range(500):
    process = exp_thinning_hawkes(1, 0.6, 1 , max_time=500)
    process.simulate()
    timestamps = process.timestamps 

    tlist+= [timestamps]

#%%


learner = estimator_unidim_multi_rep()
learner.fit(tlist)

#%%
learner.GOF_bootstrap(sup_compensator=100)
# %%
