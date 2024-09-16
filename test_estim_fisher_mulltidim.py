#%%

from functions.hawkes_process import *
from functions.GOF import *
from functions.compensator import *
from functions.multivariate_exponential_process import *
from functions.estimator_class_multi_rep import *
import scipy

np.random.seed(0)

m = np.array([0.5, 0.2]).reshape((2,1))
a = np.array([[0.4, 0.2,], 
                  [0.2, 0.3]] )
b = np.array([[1],[1.5]])



## simulation 

from functions.estimator_class import * 

param = []
hessiane = []

for k in range(100):
    
    hawkes_multi = multivariate_exponential_hawkes(m=m,
                                    a=a, 
                                    b=b,
                                    max_time=1000)
    hawkes_multi.simulate()


    learner = multivariate_estimator(dimension=2)
    param+=[learner.fit( hawkes_multi.timestamps)]
    hessiane+=[approx_fisher_muti_dim( param[-1], hawkes_multi.timestamps, max_time = True)]
    
    
#%%
theta = np.concatenate((m.flatten(), a.flatten(), b.flatten() ), axis =0)
index = 7
std_estim = [np.sqrt(np.diag(np.linalg.inv(mat))) for mat in hessiane]
stat =  np.sqrt(700)*(np.array(param)[:,index]- theta[index])/np.array(std_estim)[:,index]    

with r_inline_plot():   
    normality_test( robjects.FloatVector(stat))
# %%
