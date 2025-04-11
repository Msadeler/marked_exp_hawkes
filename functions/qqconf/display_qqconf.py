#%%
from functions.qqconf.get_qq_band import *


def qq_conf_test(obs, distribution ,alpha=0.05,plot=False):
  """
  Confidence interval and output of the test H0 : obs follow distribution 
  ----------------------------------------------------------------------------

  obs: list or array
    observation
  distribution : string or a ppf function
    distribution cas either be "normal","uniform", or a ppf function
  alpha : float between (0,1)
    level of error
  plot : bool
    wheter to plot the qqconf graph
  """
  
  if distribution=="normal":
    method = distribution
    qdistribution = scipy.stats.norm.ppf
  elif distribution == "uniform":
    qdistribution = scipy.stats.uniform.ppf
    method = distribution
  else :
    method = "median"
    qdistribution = distribution


  obs_pts = np.sort(obs)
  c = 0.5
  n = len(obs)
  global_bounds = get_qq_band(n,alpha,method,qdistribution)


  global_low = global_bounds['lower_bound']
  global_high = global_bounds['upper_bound']
  exp_pts = global_bounds['expected_value']
  ext_quantile = get_extended_quantile(distribution, n)

  low_exp_pt= c * qdistribution(np.asarray(ext_quantile['low_pt'])) + (1 - c) * exp_pts[0]
  high_exp_pt = c * qdistribution(np.asarray(ext_quantile['high_pt'])) + (1 - c) * exp_pts[-1]

  ypts = obs_pts

  left = exp_pts[0]
  right = exp_pts[-1]
  bottom = min(ypts)
  top = max(ypts) 

  if plot :
    fig, ax = plt.subplots()

    ax.set_ylim(left, right)
    ax.set_ylim(bottom, top)

  global_low = [float(global_low[0])] + global_low +  [float(global_low[n-1])]
  global_high = [float(global_high[0])] + global_high + [float(global_high[n-1])]
  exp_pts = [float(low_exp_pt)] + exp_pts +  [float(high_exp_pt)]

  if plot :
    ax.fill_between(exp_pts , global_low , global_high, alpha=0.2, color = 'deepskyblue')
    ax.scatter( exp_pts[1:-1],obs_pts, c='black', alpha=0.9,  s=[5]*len(obs_pts))
    ax.plot([min(exp_pts), max(exp_pts)], [min(exp_pts), max(exp_pts)])
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')


  
  if sum(np.array( obs_pts)< np.array(global_low[1:n+1])) + sum( np.array( obs_pts)>np.array(global_high[1:n+1]))  ==0:
    return(True)
  else :
    return(False)
  
# %%
