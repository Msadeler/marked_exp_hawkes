import numpy as np
import scipy

def F(mark,t,psi):
    return(1- np.exp( -psi*mark ))

def f(t,mark, psi):
    return(psi*np.exp(-psi*mark))

def phi(mark, gamma, psi):
    return((psi-gamma)/psi*np.exp(mark*gamma))

def phi1(mark, gamma, psi):
    return(psi**(gamma)/scipy.special.gamma(gamma+1)*mark**(gamma) )
