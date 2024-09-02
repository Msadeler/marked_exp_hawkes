import numpy as np
from scipy import stats
from scipy.optimize import minimize
from functions.likelihood_functions import *
from functions.GOF import *


    

class estimator_unidim_daichan(object):
    """
    Estimator class for Non-marked Exponential Hawkes process obtained through minimizaton of a loss using the L-BFGS-B algorithm.
    mu and beta are known

    """

    def __init__(self, 
                 mu,
                 beta,
                 loss=likelihood_daichan, 
                 a_bound = None, 
                 initial_guess=np.array((0.0)), 
                 options={'disp': False}):
        
        """
        Parameters
        ----------
        loss : {loglikelihood, likelihood_approximated} or callable.
            Function to minimize. Default is loglikelihood.
        
        a_bound: None or True
            Wheter of not to consider inhibition. If None, inhibition is autorised, otherwise, inhibition is prohibed
            Default is None

            
        initial_guess : array of float.
            Initial guess for estimated parameters. 
            Default is np.array((1.0, 0.0, 1.0)).
            
            
        options : dict
            Options to pass to the minimization method. Default is {'disp': False}.
            
        """
          

        self.options = options
        self.beta = beta
        self.m = mu
        self.bounds = [(a_bound, None)]
        self.initial_guess = [0.0]
        self.loss = loss

            
            

    def fit(self, timestamps:list , markList=[]):
        
        """
        Parameters
        ----------
        timestamps : list of float
            Ordered list containing event times.
            
            
        markList : list of float
            List containing the value of the mark for each event of the list timestamps
            This argument is necessery if mark = True 
               
        """
        
        
        self.time_jump = timestamps

        

        arguments = (self.m, self.beta, timestamps)
            

    
        self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",
                            args=arguments, bounds=self.bounds,
                            options=self.options)
            
        
        self.a_estim = self.res.x
        
        self.fit = True
        
        return(self.res.x)
    
    
    def time_change(self):
        
 
        self.transform_time = time_change_unidim_diff(self.a_estim, self.time_jump)
            
            
        return(self.transform_time)
    
    
      
class loglikelihood_estimator(object):
    """
    Estimator class for Exponential Hawkes process obtained through minimizaton of a loss using the L-BFGS-B algorithm.

    Attributes
    ----------
    res : OptimizeResult
        Result from minimization.
        
    
    """

    def __init__(self, 
                 loss=loglikelihood, 
                 a_bound = None, 
                 initial_guess=np.array((1.0, 0.0, 1.0)), 
                 options={'disp': False}, 
                 mark=False, 
                 name_arg_f=[], 
                 name_arg_phi=[], 
                 f=lambda x,t: 1,
                 phi=lambda x:1,
                 initial_guess_f = [], 
                 initial_guess_phi = [], 
                 bound_phi = [],
                 bound_f = [],
                 bound_beta = None):
        
        """
        Parameters
        ----------
        loss : {loglikelihood, likelihood_approximated} or callable.
            Function to minimize. Default is loglikelihood.
        
        a_bound: None or True
            Wheter of not to consider inhibition. If None, inhibition is autorised, otherwise, inhibition is prohibed
            Default is None

            
        initial_guess : array of float.
            Initial guess for estimated parameters. 
            Default is np.array((1.0, 0.0, 1.0)).
            
            
        options : dict
            Options to pass to the minimization method. Default is {'disp': False}.
            
        mark: bool
            Whether to consider a mark. Default is False
            
        f: density function
            function of (t, mark) that give the density of the mark at time t
        
        phi: function
            impact's function of the neuron on the process. This function must be strictly positive.
            
        name_arg_f : list 
            dict containing the name of the arguments of f 
            
        name_arg_phi : list 
            dict containing the name of the arguments of phi  
            
            
        initial_guess_f: list 
            contains a list of the initial value for each parameters of f. 
            This values will be use to initiate the minimization of log-lik
            
        initial_guess_phi  list 
            contains a list of the initial value for each parameters of phi. 
            This values will be use to initiate the minimization of log-lik
            
        bound_f : list
            list of bound, one for each parameters of f. 
            This list is use to constriain the value of those parameters during log-lik minimization
            
        bound_phi : list
            list of bound, one for each parameters of phi. 
            This list is use to constriain the value of those parameters during log-lik minimization
            
        
        """
  
        if ( mark and  ((len(bound_phi) + len(name_arg_phi) + len(bound_f)+len(name_arg_f)+ len(initial_guess_f)+ len(initial_guess_phi)==0) and phi is None and f is None) ):
            raise ValueError(" Mark is true but no parameters given for density or impact function")

            
        if not ( ( len(initial_guess_f)==len(bound_f) )  and ( len(name_arg_f)==len(bound_f) ) and  ( len(initial_guess_f) == len(name_arg_f) ) ):
             raise ValueError(" Issu with argument of the density function: one among initial_guess_f ,bound_f or  name_arg_f containt too much or not enougth argument")       
             
             
             if not ( len(initial_guess_phi) == len(bound_phi) )  and (len(name_arg_phi)== len(bound_phi) ) and  (len(initial_guess_phi) == len(name_arg_phi)):
                 raise ValueError(" Issu with argument of the impact function:  one among initial_guess_phi,bound_phi or  name_arg_phi containt too much or not enougth argument")     
        
        self.mark = mark
        self.options = options
        self.name_arg_f = name_arg_f
        self.name_arg_phi = name_arg_phi
        self.f = f
        self.phi = phi
        self.bound_beta = bound_beta

        if self.mark: 
            
            self.loss = loglikelihoodMarkedHawkes
            self.bounds = bound_f + bound_phi+  [(0.0, None), (a_bound, None), (0.0,bound_beta)]
            self.initial_guess = initial_guess_f+ initial_guess_phi + [1.0, 0.0, 1.0]


        else: 
            self.bounds = [(0.0, None), (a_bound, None), (0.0, bound_beta)]
            self.initial_guess = [1.0, 0.0, 1.0]
            self.loss = loss

            
            

    def fit(self, timestamps:list):
        
        """
        Parameters
        ----------
        timestamps : list of float
            Ordered list containing event times.
            
            
        markList : list of float
            List containing the value of the mark for each event of the list timestamps
            This argument is necessery if mark = True 
               
        """
        
        
        self.time_jump = timestamps
        arguments  = (self.time_jump, self.name_arg_f, self.name_arg_phi, self.f, self.phi)
            
    
        self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",
                            args=arguments, bounds=self.bounds,
                            options=self.options)
            
        

            
        self.arg_f_estim = dict(zip(list(self.name_arg_f) ,self.res.x[:len(self.name_arg_f)]))
        self.arg_phi_estim = dict(zip(list(self.name_arg_phi),self.res.x[len(self.name_arg_f) :len(self.name_arg_f)+len(self.name_arg_phi)]))
        self.theta_estim = self.res.x[len(self.name_arg_f)+len(self.name_arg_phi):]
        
        self.fit = True
        
        return(self.res.x)
    
    
    def time_change(self):
        
        
        if self.mark:
            
            self.transform_time = time_change_mark_unidim_diff(self.theta_estim, self.time_jump, self.mark_list, self.phi, self.f, self.arg_f_estim, self.arg_phi_estim)

        else: 
            self.transform_time = time_change_unidim_diff(self.theta_estim, self.time_jump)
            
            
        return(self.transform_time)
    
    
    def mark_change(self,  cdf = None, borne_inf = -np.inf):
        
        
        """
        Parameters
        ----------
        cdf : the cumulative density function associated to the density f of the marked process
            cdf must have the same argument as f
            If cdf is None, the cumulative distribution is computed using integrate.quad of the density f
         
        borne_inf: When cdf is None, borne_inf can be specify to make sure the cumulative distribution is computed on the right interval
            
            
        Return
        ----------   
        List of float corresponding to the transformation of mark    
              
        """
                    
        if not self.mark:
            raise ValueError("Le processus a été définis comme non marqué")
            
            
        if cdf is None:

            self.transform_mark = mark_change_mark_unidim(self.time_jump, self.mark_list, self.f, self.arg_f_estim, borne_inf)
        
        else: 
            self.transform_mark = cdf(self.mark_list, self.tList, **arg_f)
        return(self.transform_mark)
            

        
        
        if self.mark:
            
            self.transform_time = time_change_mark_unidim(self.theta_estim, self.time_jump, self.mark_list, self.phi, self.f, self.arg_f_estim, self.arg_phi_estim)

        else: 
            self.transform_time = time_change_unidim(self.theta_estim, self.time_jump)
            
            
        return(self.transform_time)
    
    
    def mark_change(self,  cdf = None, borne_inf = -np.inf):
        
        
        """
        Parameters
        ----------
        cdf : the cumulative density function associated to the density f of the marked process
            cdf must have the same argument as f
            If cdf is None, the cumulative distribution is computed using integrate.quad of the density f
         
        borne_inf: When cdf is None, borne_inf can be specify to make sure the cumulative distribution is computed on the right interval
            
            
        Return
        ----------   
        List of float corresponding to the transformation of mark    
              
        """
                    
        if not self.mark:
            raise ValueError("Le processus a été définis comme non marqué")
            
            
        if cdf is None:

            self.transform_mark = mark_change_mark_unidim(self.time_jump, self.mark_list, self.f, self.arg_f_estim, borne_inf)
        
        else: 
            self.transform_mark = cdf(self.mark_list, self.tList, **arg_f)
        return(self.transform_mark)
            
    

class multivariate_estimator(object):
    """
    Estimator class for Exponential Hawkes process obtained through minimizaton of a loss using the L-BFGS-B algorithm.

    Attributes
    ----------
    res : OptimizeResult
        Result from minimization.
    """

    def __init__(self,
                 loss=multivariate_loglikelihood, 
                 dimension=None, 
                 initial_guess="random",
                 options=None, 
                 a_bound = None, 
                 beta_bound = None,
                 interaction_coeff = None, 
                 mark=False, 
                 name_arg_f={}, 
                 name_arg_phi={}, 
                 f=lambda x,t: 1,
                 phi=lambda x:1,
                 initial_guess_f = [], 
                 initial_guess_phi = [], 
                 bound_phi = [],
                 bound_f = []):
        """
        Parameters
        ----------
        loss : {loglikelihood, likelihood_approximated} or callable.
            Function to minimize. Default is loglikelihood.
            
        dimension : int
            Dimension of problem to optimize. Default is None.

        initial_guess : array of float.
            Initial guess for estimated parameters. 
            Default is np.array((1.0, 0.0, 1.0)).
            
            
        options : dict
            Options to pass to the minimization method. Default is {'disp': False}.
        
        a_bound: None or a real number
            If a_bound if None, all kind of interaction are search. 
            If not, the minimisation function search for a_ij in the line (a_bound, +infinity)
            Default is None
        
        
        interaction_coeff: None or a array of shape (dimension, dimension)
        
            If interaction_coeff is an array, each index (i,j) of the array such that (interaction_coeff)_ij = 0 
            will induce a coefficient a_ij = 0 by the algorithm
            If interaction_coeff is None, all a_ij are computed using log-lik maximization
            Default is None
            
            
        mark: bool
            Whether to consider a mark. 
            Default is False.
            
        f: density function
            function of (t, mark) that give the density of the mark at time t.
            Default is the constant function equal to 1.
        
        phi: function
            Impact's function of the neuron on the process. 
            This function must return a matrix of shape (dimension, dimension) and must be strictly positive.
            Default is the constant function equal to 1.
            
            
        name_arg_f : list 
            dict containing the name of the arguments of f 
            
        name_arg_phi : list 
            dict containing the name of the arguments of phi  
            
            
        initial_guess_f: list 
            contains a list of the initial value for each parameters of f. 
            This values will be use to initiate the minimization of log-lik
            
        initial_guess_phi  list 
            contains a list of the initial value for each parameters of phi. 
            This values will be use to initiate the minimization of log-lik
            
        bound_f : list
            list of bound, one for each parameters of f. 
            This list is use to constriain the value of those parameters during log-lik minimization
            
        bound_phi : list
            list of bound, one for each parameters of phi. 
            This list is use to constriain the value of those parameters during log-lik minimization
                      
        


        """
                
        
        if dimension is None:
            raise ValueError("Dimension is necessary for initialization.")
            
        self.loss = loss
        self.dim = dimension
        self.mark = mark
        self.f = f
        self.phi = phi 
        self.name_arg_f = name_arg_f
        self.name_arg_phi = name_arg_phi
        self.bound_f = bound_f
        self.bound_phi = bound_phi
        self.initial_guess_phi = initial_guess_phi
        self.initial_guess_f = initial_guess_f
        self.nb_arg_phi = 0
        self.beta_bound = beta_bound

        
        self.bounds = [(1e-12, None) for i in range(self.dim)] + [(a_bound, None) for i in range(self.dim * self.dim)] + [
            (1e-12, beta_bound) for i in range(self.dim)]
        
                
        ### Where there is no interaction asked the coeff are contrained to [0,0]
        if interaction_coeff is None:
            self.interaction_coeff = np.ones((self.dim,self.dim)).flatten()
        else: 
            self.interaction_coeff = interaction_coeff.flatten()

        interaction = [(0,0) if self.interaction_coeff[i]==0 else self.bounds[self.dim+i] for i in range(len(self.interaction_coeff))]
        self.bounds[self.dim:self.dim * (self.dim + 1)] = interaction
        

    
        if isinstance(initial_guess, str) and initial_guess == "random":
            self.initial_guess =  [1]*self.dim + [0]*self.dim**2  + [1]*self.dim
            
        ## add the bound for the coeff of f and phi if the processed is marked
        
        
        if self.mark:     
            self.loss = opimisation_marked_hawkes
            self.bounds =  bound_f + bound_phi+  self.bounds
            self.initial_guess = np.array(initial_guess_f+ initial_guess_phi + self.initial_guess)         
            self.nb_arg_phi = len(bound_phi)
        
        if options is None:
            self.options = {'disp': False}
        else:
            self.options = options
            
        self.fitted = False
            

    def fit(self, timestamps,limit=1000):
        
        """
        Parameters
        ----------
        timestamps : list of tuple.
            Ordered list containing event times and marks.
            
        """


        self.time_jump = timestamps
        
        if self.mark:
            argument = (self.time_jump, self.name_arg_phi,self.name_arg_f, self.phi,self.f, self.nb_arg_phi, self.dim)
        else: 
            argument = (self.time_jump)

        self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B", 
                args=argument,
                bounds=self.bounds,
                options=self.options)
        
        self.arg_f_estim = dict(zip(list(self.name_arg_f) ,self.res.x[:len(self.name_arg_f)]))
        self.arg_phi_estim = dict(zip(list(self.name_arg_phi),self.res.x[len(self.name_arg_f) :len(self.name_arg_f)+len(self.name_arg_phi)]))
        self.theta_estim = self.res.x[len(self.name_arg_f)+len(self.name_arg_phi):]
        
        self.fitted = True 


        return self.res.x
        
    
    
    def time_change(self):
       
        if not self.fitted:
            raise ValueError("Estimator need to be fitted first")
    
        if self.mark:

            self.transform_time = time_change_mark_multidim(theta = self.theta_estim, 
                                                               tList = self.time_jump, 
                                                               phi = self.phi,
                                                               arg_phi = self.arg_phi_estim)
        else: 
            self.transform_time = time_change_multidim(theta = self.theta_estim, tList = self.time_jump)
            
        return(self.transform_time)
   
    def mark_change(self,  cdf = None, borne_inf = -np.inf):
       
      
       """
       Parameters
       ----------
       cdf : the cumulative density function associated to the density f of the marked process
           cdf must have the same argument as f
           If cdf is None, the cumulative distribution is computed using integrate.quad of the density f
        
       borne_inf: When cdf is None, borne_inf can be specify to make sure the cumulative distribution is computed on the right interval
           
           
       Return
       ----------   
       List of float corresponding to the transformation of mark    
             
       """
       if not self.mark:
           raise ValueError("The process was defined as unmarked")

       if not self.fitted:
           raise ValueError("Estimator need to be fitted first")           
           
       if cdf is None:

           self.transform_mark = mark_change_mark_unidim(self.time_jump, self.mark_list, self.f, self.arg_f_estim, borne_inf)
       
       else: 
           self.transform_mark = cdf(self.mark_list, self.tList, **arg_f)
       return(self.transform_mark)

