#%%
from rpy2 import robjects
import matplotlib.pyplot as plt
from rpy2.robjects.packages import importr


rqqconf = importr('qqconf')


robjects.r('''bounds_computation <- function(n, alpha){
  return(get_bounds_two_sided(n=n, alpha = alpha))
}''')

get_bounds_two_sided = robjects.globalenv['bounds_computation']


"""
def get_level_from_bounds_two_sided(lower_bounds,
                                            upper_bounds):
    return()


def get_bounds_two_sided(n, alpha, max_it =100, tol=1e-8):

    if n ==1 :

        return({'lower_bound' : [alpha / 2],
            'upper_bound' : [1 - alpha / 2],
            'x ':  0.5,
            'local_level' : alpha} )
    
    elif n<1:
        print('n must be stricly positiv')

    else:
        n_param = n
        if (n >= 10):
            
            # Approximation only available for n < 10
            method = "best_available"
            
        else :
            
            method =- "search"
            alpha_epsilon = 10 ^ (-5)
  
        # Approximations are only available for alpha = .05 or alpha = .01
            if (method == "search") :
                
                eta_high = alpha
                eta_low = alpha / n
                eta_curr = eta_low + (eta_high - eta_low) / 2
                n_it = 0
                
                while (n_it < max_it) :
                
                    n_it = n_it + 1
                    h_vals = list(map( lambda x,y : scipy.stats.beta.ppf(eta_curr, x,y), np.arange(1,n+1),  np.arange(n,0, -1)))
                    g_vals = list(map( lambda x,y : scipy.stats.beta.ppf(1 - (eta_curr / 2), x,y), np.arange(1,n+1),  np.arange(n,0, -1)))
                
                    test_alpha = get_level_from_bounds_two_sided(h_vals, g_vals)
                    
                    if (abs(test_alpha - alpha) / alpha <= tol) :
                        break
                    
                    if (test_alpha > alpha) :
                        eta_high = eta_curr
                        eta_curr = eta_curr - (eta_curr - eta_low) / 2
                        
                    elif (test_alpha < alpha) :
                        eta_low = eta_curr
                        eta_curr = eta_curr + (eta_high - eta_curr) / 2
                        
                    
                    
                    eta = eta_curr

                if(n_it == max_it):
                    print("Maximum number of iterations reached.")

            
                    
"""                       
        
            
# %%
