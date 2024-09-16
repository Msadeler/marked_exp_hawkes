

import numpy as np 

def compute_hessian_marked(theta,tList, Tmax = None):

    psi,gamma,m,a,b = theta

    normal_coeff = (psi-gamma)/psi


    partial_lambda = np.array([0,0,1,0,0], dtype=float)
    lambda_Tk = m


    last_time, last_mark = tList[1]

    fisher_approx = np.outer(partial_lambda,partial_lambda)/(lambda_Tk**2)

    for time, mark in tList[2:-1]:

        delta_t = time-last_time
        
        
        partial_lambda[0] = np.exp(-b*delta_t)*(partial_lambda[0] +gamma*a*np.exp( gamma*last_mark )/(psi**2) ) ## partial_psi lambda(Tk)
        
        
        partial_lambda[1] = np.exp(-b*delta_t)*(partial_lambda[1] + a*np.exp( gamma*last_mark ) *(normal_coeff*last_mark-1/psi ) ) ## partial_gamma lambda(Tk)
        
        partial_lambda[3] = np.exp(-b*delta_t)*(partial_lambda[3]+normal_coeff*np.exp(gamma*last_mark)) ## partial_a lambda(Tk)
        
        partial_lambda[4] = np.exp(-b*delta_t)*(partial_lambda[4] - (  lambda_Tk+ a*normal_coeff*np.exp(gamma*mark)-m)*delta_t ) ## partial_b lambda(Tk)
        
        lambda_Tk = m+ np.exp( -b*delta_t)*( lambda_Tk + a*np.exp(gamma*last_mark)*normal_coeff-m)

        vector_jump = partial_lambda/lambda_Tk

        fisher_approx += np.outer( vector_jump, vector_jump)
        
        last_time, last_mark = time,mark
        
    if Tmax is None:
        return(fisher_approx/time)
    else: 
        return(fisher_approx/Tmax)