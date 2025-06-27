#%%

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import functions as hp
from functions.paramtrised_function import *


data_earthquake = pd.read_csv('data/ogata/ogata.csv', index_col=0).loc[:, ['time', 'magnitude']]
data_tuple = [(0,0)] +list(data_earthquake.itertuples(index=False, name = None)) +  [(800,0)]  ### put data here

def rec_formula(t, m,a,b,arg_f,arg_phi, phi, timebefore, intensity_last_jump, time_transformed):
    return( time_transformed+ m*(t-timebefore[0])+ (1-np.exp(-b*(t-timebefore[0])))*(intensity_last_jump+a*phi(timebefore[1], **arg_phi, **arg_f)-m)/b )

rec_formula = rec_formula


bval = 500

estimator = hp.estimator_bootstrap(mark=True,
                            name_arg_f=['psi'], 
                                name_arg_phi=['gamma'], 
                                f=f,
                                F=F,
                                phi=phi,
                                initial_guess_f = [1], 
                                initial_guess_phi = [0], 
                                bound_phi = [(1e-5, None)],
                                bound_f = [(1e-5,None)]
                                )
estimator.boostrap_procedure( tlist =data_tuple, B=bval,max_time=True,rec_formula=rec_formula)

#%%
result = estimator.test_one_coeff(coefficient_index=1, value=0)

print('pvalue:', 2*(1-scipy.stats.norm.cdf(abs(result['stat']))))
# %%
