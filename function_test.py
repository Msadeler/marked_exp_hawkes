import numpy as np

def F(mark,t,psi):
    return(1- np.exp( -psi*mark ))

def phi(mark, gamma, psi):
    return((psi-gamma)/psi*np.exp(mark*gamma))

def f(t,mark, psi):
    return(psi*np.exp(-psi*mark))


def phi1(mark, gamma, mu, sigma):
    return(np.exp(gamma*(mark-mu) - mu**2*sigma**2/2 ) )

def density(t,mark, mu, sigma):
    return(1/np.sqrt(2*np.pi*sigma**2)*np.exp( - (mark-mu)**2/(2*sigma**2)))