#%%
from functions.qqconf.get_qq_band import *
import scipy

def qq_conf_multi(obs, distribution ,alpha=0.05,plot=False, label = None, color=None, marker=None, size = 10, return_plot = False):
    """
    Plot several empirical distribution to compare it the same theoretical distribution
    Warning: each empirical distribution must have the same size 
    """


    if len(np.unique([len(x) for x in obs]))>1 :
        print("Err : all empirical distribution must have the same size in order to be compared")
    else: 

        if distribution=="normal":
            method = distribution
            qdistribution = scipy.stats.norm.ppf
        elif distribution == "uniform":
            qdistribution = scipy.stats.uniform.ppf
            method = distribution
        else :
            method = "median"
            qdistribution = distribution


        obs_pts = [np.sort(np.array(x).flatten()) for x in obs]
        c = 0.5
        n = len(obs[0])

        conf_int = 1 - alpha
        conf = [alpha / 2, conf_int + alpha / 2]

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
        bottom = np.array(ypts).min()
        top = np.array(ypts).max()

        if plot :
            fig, ax = plt.subplots()

            ax.set_ylim(left, right)
            ax.set_ylim(bottom, top)

        pointwise_low = list(map(lambda x,y : scipy.stats.norm.ppf(scipy.stats.beta.ppf(conf[0],x,y) ),np.arange(1, n+1), np.arange(n, 0, -1)) )
        pointwise_high = list(map(lambda x,y : scipy.stats.norm.ppf(scipy.stats.beta.ppf(conf[1],x,y) ),np.arange(1, n+1), np.arange(n, 0, -1)) )

        global_low = [float(global_low[0])] + global_low +  [float(global_low[n-1])]
        global_high = [float(global_high[0])] + global_high + [float(global_high[n-1])]

        pointwise_low =  pointwise_low[0] + pointwise_low +  pointwise_low[n-1]
        pointwise_high =  pointwise_high[0] + pointwise_high +  pointwise_high[n-1]
        exp_pts = [float(low_exp_pt)] + exp_pts +  [float(high_exp_pt)]

        output_test =[sum(np.array( empirical_distrib)< np.array(global_low[1:n+1])) + sum( np.array( empirical_distrib)>np.array(global_high[1:n+1]))  ==0 for empirical_distrib in ypts ]

        if plot :
            ax.fill_between(exp_pts , global_low , global_high, alpha=0.2, color = 'deepskyblue')
            ax.plot([min(exp_pts), max(exp_pts)], [min(exp_pts), max(exp_pts)])
            
            if (not label) or ( not isinstance(label, list)) or ( isinstance(label, list) and len(label)!= len(ypts)):
                print("Wrong format for labels")
                label = [f"EmpDistrib{i}" for i in range(len(ypts))]
                plot_label = False
            else : 
                plot_label=True

            if (not color) or ( not isinstance(color, list)) or ( isinstance(color, list) and len(color)!= len(ypts)):
                print("Default color taken")
                color = [f"C{i+1}" for i in range(len(ypts))]

            if (not marker) or ( not isinstance(color, list)) or ( isinstance(color, list) and len(color)!= len(ypts)):
                print("Default marker taken")
                color = [f"{i}" for i in range(len(ypts))]


            for i,empirical_distrib in enumerate(ypts):
                ax.scatter( exp_pts[1:-1],empirical_distrib, alpha=0.9,  s=[size]*n, label=str(label[i]), color = color[i], marker = marker[i])
            if plot_label:
                ax.legend()
                
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel('Sample Quantiles')

        if return_plot and plot :
            return(output_test, (fig,ax))
        else :
            return(output_test)


# %%
