#%%
from functions.likelihood_functions import *
from functions.hawkes_process import *
from functions.GOF import *
from functions.compensator import *
from functions.multivariate_exponential_process import *
from functions.estimator_class_multi_rep import *
import scipy
from functions.estimator_class import *

m,a,b = 1, 1, 2

from functions.paramtrised_function import *


hawkes = exp_thinning_hawkes_marked(m=m,
                                        a=a, 
                                        b=b, 
                                        mark_process=True,
                                        F=F1, 
                                        arg_F={'psi':2}, 
                                        phi=phi, 
                                        arg_phi={'gamma':1}, 
                                        max_time=1000)
hawkes.simulate()




# %%

learner = loglikelihood_estimator(mark=True, 
                                     f=f, 
                                    name_arg_f=['psi'], 
                                    phi=phi, 
                                    name_arg_phi=['gamma'], 
                                    bound_f=[(1e-5, None)], 
                                    bound_phi=[(None,None)], 
                                    initial_guess_f=[1], 
                                    initial_guess_phi=[0])


learner.fit(hawkes.timestamps, max_time = True)

# %%
