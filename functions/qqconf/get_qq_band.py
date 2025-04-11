import scipy
from functions.qqconf.utils import *
from functions.qqconf.get_bound_two_sided import *

def get_qq_band(n=None,
        alpha = 0.05,
        method = "normal",
        qdistribution = None):
    if not n:
        print( "n must be supplied")

    if method == "normal":
        raw_pts = ppoints(n)
    elif method == "uniform":
        raw_pts = ppoints(n,0)
    elif method =="median":
        raw_pts = list(map( lambda x,y : scipy.stats.beta.ppf( 0.5, x,y),np.arange(1, n+1), np.arange(n, 0, -1)))


    exp_pts = list(map( qdistribution, raw_pts))
    ell_bounds = get_bounds_two_sided(n,alpha)

    lower_bound = np.asarray(ell_bounds[0])
    upper_bound = np.asarray(ell_bounds[1])

    
    lower_bound = list(map(qdistribution,lower_bound))
    upper_bound = list(map(qdistribution,  upper_bound))

    return( 
      {'lower_bound': lower_bound,
      'upper_bound' : upper_bound,
      'expected_value' : exp_pts}
    )
  