#%%
import scipy
import numpy as np

between = lambda x,gte, lte : x>=gte & x <= lte 

def ppoints(vector,a=None):
    '''
    Mimics R's function 'ppoints'.
    '''

    if isinstance(vector, (float, int)):
      vector = [vector]

    m_range = int(vector[0]) if len(vector)==1 else len(vector)
    n = vector[0] if len(vector)==1 else len(vector)
    if a is None:
      a = 3./8. if n <= 10 else 1./2
    m_value =  n if len(vector)==1 else m_range
    pp_list = [((m+1)-a)/(m_value+(1-a)-a) for m in range(m_range)]
    return pp_list

def get_extended_quantile(exp_pts_method, n):

  if (exp_pts_method == "uniform"):
    high_pt = 1 - 1/max( n+2, n*1.25)
    low_pt = 1/max( n+2, n*1.25)
  elif (exp_pts_method == "normal"):
    new_samp_size = np.floor(n * 1.3)
    ppoints_adj = ppoints(new_samp_size)
    low_pt = ppoints_adj[0]
    high_pt = ppoints_adj[-1]
  else:
    new_samp_size = np.floor(n * 1.02)
    ppoints_adj = ppoints(new_samp_size)
    low_pt = ppoints_adj[0]
    high_pt = ppoints_adj[-1]

  return({'low_pt':low_pt, 'high_pt':high_pt})


# %%
