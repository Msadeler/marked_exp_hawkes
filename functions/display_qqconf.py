import rpy2.robjects as robjects
from contextlib import contextmanager
from rpy2.robjects.lib import grdevices
from IPython.display import Image, display
from rpy2 import robjects
from rpy2.robjects.packages import importr

rqqconf = importr('qqconf')
rstats = importr('stats')

robjects.r('''
           uniformity_test <- function(pvals, alpha = 0.05, min = 0, max = 1)
           {
           qq_conf_plot(pvals, 
           distribution = qunif, 
           alpha = alpha,
           dparams = list('min' =0,'max'=1), 
           polygon_params = list( col='powderblue', border = FALSE),
           points_params = list(cex=0.2)
           )}
           ''')


robjects.r('''
           normality_test <- function(pvals, alpha = 0.05)
           {
           qq_conf_plot(pvals, 
           distribution = qnorm, 
           alpha = alpha,
           dparams = list('mean' =0,'sd'=1), 
           polygon_params = list( col='powderblue', border = FALSE),
           points_params = list(cex=0.2)
           )}
           ''')


uniformity_test = robjects.globalenv['uniformity_test']
normality_test = robjects.globalenv['normality_test']


@contextmanager
def r_inline_plot(width=600, height=600, dpi=100):

    with grdevices.render_to_bytesio(grdevices.png, 
                                     width=width,
                                     height=height, 
                                     res=dpi) as b:

        yield

    data = b.getvalue()
    display(Image(data=data, format='png', embed=True))