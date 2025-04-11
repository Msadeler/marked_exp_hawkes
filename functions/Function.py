import numpy as np
from functions.likelihood_functions import *
from functions.estimator_class import *
from functions.hawkes_process import *
from functions.TimeChange import *
from functions.multivariate_exponential_process import *
import time
import scipy



mu, alpha, beta =1,0.6,2
Tmax = 5000
phi_arg = 0.5


def F(mark,time,a):
    return(scipy.stats.expon.cdf(mark, scale = 1/2))

def phi(mark, s, a):
    return((a-s)/a*np.exp(mark*s))



def phi1(mark, s,a):
    return((a**(s)/scipy.special.gamma(s+1))*mark**s)


def f(mark,t, a):
    return(a*np.exp(-a*mark))

