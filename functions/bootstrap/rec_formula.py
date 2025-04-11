import numpy as np

def rec_compensator_unidim_MEHP(t, m,a,b,arg_f,arg_phi, phi, timebefore, intensity_last_jump, time_transformed):
    return( time_transformed+ m*(t-timebefore[0])+ (1-np.exp(-b*(t-timebefore[0])))*(intensity_last_jump+a*phi(timebefore[1], **arg_phi, **arg_f)-m)/b )


def rec_compensator_multidim_EHP(t, mu, alpha, beta, Tk, lambdaTk, LambdaTk):
    t_star = Tk + np.log( 1 -  np.minimum(lambdaTk + alpha, 0)/mu)/beta
    compensator = (t_star < t)*(np.multiply(mu,t-t_star)) + np.multiply(lambdaTk + alpha-mu, np.exp(-beta*(t_star- Tk))- np.exp(-beta*(t-Tk)))/beta
    
    return(compensator + LambdaTk)