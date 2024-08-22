import numpy as np
from functions.likelihood_functions import *
from functions.GOF import *
from functions.compensator import *
import multiprocessing
import functools
from functions.display_qqconf import *
import matplotlib.pyplot as plt


class estimator_unidim_multi_rep(object):
    
    def __init__(self,
                loss = loglikelihood,
                a_bound = None, 
                initial_guess=[1.0, 0.0, 1.0], 
                options={'disp': False},
                mark=False,
                name_arg_phi=[], 
                name_arg_f=[],
                f=lambda x,t: 1,
                phi=lambda x:1,
                initial_guess_f = [], 
                initial_guess_phi = [], 
                bound_phi = [],
                bound_f = [],
                bound_b = None
                ):

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
                
        self.mark = mark
        self.options = options
        self.name_arg_f = name_arg_f
        self.name_arg_phi = name_arg_phi
        self.f = f
        self.phi = phi
        self.bound_b = bound_b

        if self.mark: 
            
            self.loss = loglikelihoodMarkedHawkes
            self.bounds = bound_f + bound_phi+  [(0.0, None), (a_bound, None), (0.0,bound_b)]
            self.initial_guess = initial_guess_f+ initial_guess_phi + initial_guess


        else: 
            self.bounds = [(1e-5, None), (a_bound, None), (0.0, bound_b)]
            self.initial_guess = initial_guess
            self.loss = loss

        

    def fit(self, timestamps ,nb_cores=None):
        
        """
        Fit the Unidim Hawkes model to a list of realisations

        Parameters
        ----------
        timestamps : list containing lists of tuples 
            List of ordered list containing event times, component and mark value.

        nb_cores : int or None
            Number of core to use during the estimation
               
        """
        
        
        self.time_jump = timestamps

        if not nb_cores: 
            nb_cores = multiprocessing.cpu_count()-1 
        
        if not self.mark :
            pool = multiprocessing.Pool(nb_cores)                         
            results = pool.map(functools.partial(minimization_unidim_unmark,loss=self.loss, initial_guess=self.initial_guess, bounds=self.bounds, options=self.options) , self.time_jump)
            pool.close()

        else: 

            pool = multiprocessing.Pool(nb_cores)                         
            results = pool.map( functools.partial(minimization_unidim_marked,loss=self.loss, initial_guess=self.initial_guess,name_arg_f=self.name_arg_f,name_arg_phi=self.name_arg_phi,f=self.f, phi=self.phi,bounds=self.bounds,options=self.options) , self.time_jump)
            pool.close()
            

        self.res_list = results
            
        self.params_estim = np.array([res.x for res in self.res_list])
        self.mean_MLE = np.array(self.params_estim).mean(axis=0)
        self.mean_theta = self.mean_MLE[-3:]
        
        if self.mark:
            self.mean_f_arg = dict( zip( self.name_arg_f, self.mean_MLE[:len(self.name_arg_f)]))
            self.mean_phi_arg = dict(zip(self.name_arg_phi,
                                         self.mean_MLE[len(self.name_arg_f) :len(self.name_arg_f)+len(self.name_arg_phi)]))

        else : 
            self.mean_f_arg = {}
            self.mean_phi_arg = {}
                    
        self.fit = True
        return(self.mean_MLE)
    

    def GOF_bootstrap(self, 
                      sup_compensator=None,
                      SubSample_size= None, 
                      Nb_SubSample = 50,
                      nb_cores = -1, 
                      compensator_func = unidim_EHP_compensator, 
                      plot = True):
        

        """
        Perform the GOF procedure 


        Parameters
        ----------

        sup_compensator : float or None, default value is None
            Parameter that is, a.s, less than the mean size of the interval of realisation of the cumulated process. If not value is given, sup_compensator is equal to 0.9 times the sum of the values Lambdai(Tmax)

          
        SubSample_size : int or None
            Size of the subsamples used to perform the procedure. If None, SubSample_size set to n**(2/3) with n the number of realisation available.
            
        Nb_SubSample : int or None
            Number of time the bootstrap procedure is done. Set to 500 by default.

        nb_cores: int default value is -1
            Number of core to use, if the value, is -1, all cores are used

        compensator_func : Function 
            Compensator associated to the tested model.  
        

        """
                
        
        sample_size = len(self.time_jump)

        if nb_cores==-1: 
            nb_cores = multiprocessing.cpu_count()
        
        if not SubSample_size:
            SubSample_size = int(sample_size**(2/3))


        ## apply time transformation to all realisation 
        pool = multiprocessing.pool.ThreadPool(nb_cores)     
        time_transformed = pool.map(functools.partial(compensator_func,theta=self.mean_theta, phi=self.phi, arg_f=self.mean_f_arg, arg_phi=self.mean_phi_arg), self.time_jump)
        pool.close()
        
        ### select a subsample 
        subsample = [np.random.choice(np.arange(sample_size), size=SubSample_size, replace=False) for l in range(Nb_SubSample)]
        subsample_times = [[time_transformed[k] for k in index] for index in subsample]
        ## foor each subsample, perform gof procedure by aggegating transformed process and compute
        ## the associated pval
        
        
        pool = multiprocessing.pool.ThreadPool(nb_cores)     
        pval_list = pool.map(functools.partial(GOF_procedure,sup_compensator=sup_compensator), subsample_times)
        pool.close()


        KS_test = kstest(pval_list, cdf = 'uniform')
        
         ## display qqconf plot of the pvalue
        if plot : 
            with r_inline_plot():
                uniformity_test( robjects.FloatVector(pval_list))

        return( {"pvalList": pval_list, "KStest_stat": KS_test.statistic, "KStest_pval" : KS_test.pvalue})

        
    def test_one_coeff(self, index_coeff: int, theta_star : float, plot = None):


        """
        Perform an equality test on a coefficient of the model 


        Parameters
        ----------

        index : int
            Index of the paramter, inside the list self.param_theta, that is tested. index = 0 for mu, index = 1 for a, index = 2 b and index = 4 for arg_phi
          
        theta_star : float
            Value of the true parameter to test
        
        """

        if (self.mark or theta_star<=0):
            print("No theoretical garantee associated to this test")
            
        coeff = self.params_estim[:,index_coeff]
        
        test_stat = (coeff-theta_star)/np.std(coeff)

        ks_test = kstest( test_stat, cdf='norm')
        
        if plot : 
            with r_inline_plot():
                normality_test( robjects.FloatVector(test_stat))

        return( {"estimatorList": test_stat,  "KStest_stat": ks_test.statistic, "KStest_pval" : ks_test.pvalue })

    def test_equality_coeff(self, index_1, index_2):


        """
        Perform an equality test on a coefficient of the model 


        Parameters
        ----------

        index_1, index_2 : int
            Indexes of the paramters, inside the list self.param_theta, whose equality is tested. 
        
        """

        if (self.mark) :
            print("No theoretical garantee associated to this test")

        cov_mat = np.cov(self.params_estim[:,index_1], self.params_estim[:,index_2]) 
        stat = (self.params_estim[:,index_1] - self.params_estim[:,index_2])/ np.sqrt( cov_mat[0,0]+ cov_mat[1,1]-2*cov_mat[0,1]  )
        ks_test = kstest( stat, cdf='norm')
        
        
        with r_inline_plot():
            normality_test( robjects.FloatVector(stat))
       

        return( {"estimatorList": stat,  "KStest_stat": ks_test.statistic, "KStest_pval" : ks_test.pvalue })

        


class estimator_multidim_multi_rep(object):
    
    def __init__(self,
                 loss=multivariate_loglikelihood_simplified, 
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
                 bound_f = []
                ):

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
            self.loss = multivariate_marked_likelihood
            self.bounds =  bound_f + bound_phi+  self.bounds
            self.initial_guess = np.array(initial_guess_f+ initial_guess_phi + self.initial_guess)         
            self.nb_arg_phi = len(bound_phi)
        
        if options is None:
            self.options = {'disp': False}
        else:
            self.options = options
            
        self.fitted = False
            
    def fit(self, timestamps , markList=[], nb_cores=None):
        
        """
        Fit the Unidim Hawkes model to a list of realisations

        Parameters
        ----------
        timestamps : list of list or array
            List of ordered list containing event times.
            
            
        markList : list of list of array 
            List containing the value of the mark for each event of the list timestamps
            This argument is necessery if mark = True 

        nb_cores : int or None
            Number of core to use during the estimation
               
        """
        
        self.time_jump = timestamps

        if not nb_cores: 
            nb_cores = multiprocessing.cpu_count()-1 
        
        if not self.mark :
            
            self.mark_list = timestamps
            pool = multiprocessing.Pool(nb_cores)                         
            results = pool.map(functools.partial(minimization_multidim_unmark,loss=self.loss, initial_guess=self.initial_guess, bounds=self.bounds, options=self.options, dim = self.dim) , self.time_jump)
            pool.close()

        else: 

            self.mark_list = markList
            pool = multiprocessing.Pool(nb_cores)                         
            results = pool.map( functools.partial(minimization_multidim_marked,loss=self.loss, initial_guess=self.initial_guess,name_arg_f=self.name_arg_f,name_arg_phi=self.name_arg_phi,f=self.f, phi=self.phi,bounds=self.bounds,options=self.options, dim = self.dim) , self.time_jump, self.mark_list)
            pool.close()
            

    

        self.res_list = results
            
        self.params_estim = np.array([res.x for res in self.res_list])
        self.mean_MLE = np.array(self.params_estim).mean(axis=0)
        self.mean_theta = self.mean_MLE[-3:]
        
        if self.mark:
            self.mean_f_arg = dict( zip( self.name_arg_f, self.mean_MLE[:len(self.name_arg_f)]))
            self.mean_phi_arg = dict(zip(self.name_arg_phi,
                                         self.mean_MLE[len(self.name_arg_f) :len(self.name_arg_f)+len(self.name_arg_phi)]))

        else : 
            self.mean_f_arg = {}
            self.mean_phi_arg = {}
                    
        self.fit = True
        return(self.mean_MLE)
    

    def GOF_bootstrap(self, 
                      sup_compensator,
                      SubSample_size= None, 
                      Nb_SubSample = 500,
                      nb_cores = -1, 
                      compensator_func = unidim_EHP_compensator):
        

        """
        Perform the GOF procedure 


        Parameters
        ----------

        sup_compensator : float
            Parameter that is, a.s, less than the mean size of the interval of realisation of the cumulated process   

          
        SubSample_size : int or None
            Size of the subsamples used to perform the procedure. If None, SubSample_size set to n**(2/3) with n the number of realisation available.
            
        Nb_SubSample : int or None
            Number of time the bootstrap procedure is done. Set to 500 by default.

        nb_cores: int or None
            Number of core to use 

        compensator_func : Function 
            Compensator associated to the tested model.  
        

        """
                
        
        sample_size = len(self.time_jump)

        if nb_cores==-1: 
            nb_cores = multiprocessing.cpu_count()-1 
        
        if not SubSample_size:
            SubSample_size = int(sample_size**(2/3))


        subsample = [np.random.choice([k for k in range(sample_size)], size=SubSample_size, replace=False) for l in range(Nb_SubSample)]

        results = []

        for index in subsample:
            results+=[GOF_procedure(index,theta=self.mean_theta, tList=self.time_jump, markList=self.mark_list, compensator_func=compensator_func, sup_compensator=sup_compensator, phi=self.phi, arg_f=self.mean_f_arg, arg_phi=self.mean_phi_arg)]
     

        KS_test = kstest(results, cdf = 'uniform')
        
        with r_inline_plot():
            uniformity_test( robjects.FloatVector(results))
        
        return( {"pvalList": results, "KStest_stat": KS_test.statistic, "KStest_pval" : KS_test.pvalue})

        
    def test_one_coeff(self, index_coeff, theta_star):


        """
        Perform an equality test on a coefficient of the model 


        Parameters
        ----------

        index : int
            Index of the paramter, inside the list self.param_theta, that is tested. index = 0 for mu, index = 1 for a, index = 2 b and index = 4 for arg_phi
          
        theta_star : float
            Value of the true parameter to test
        
        """

        if (self.mark or (index_coeff == 1 and theta_star<=0)):
            print("No theoretical garantee associated to this test")


        coeff = self.params_estim[:,index_coeff]
        
        test_stat = (coeff-theta_star)/np.std(coeff)

        ks_test = kstest( test_stat, cdf='norm')
        
        

        return( {"estimatorList": test_stat,  "KStest_stat": ks_test.statistic, "KStest_pval" : ks_test.pvalue })
    
    def test_equality_coeff(self, index_1, index_2):


        """
        Perform an equality test on a coefficient of the model 


        Parameters
        ----------

        index : int
            Index of the paramter, inside the list self.param_theta, that is tested. index = 0 for mu, index = 1 for a, index = 2 b and index = 4 for arg_phi
          
        theta_star : float
            Value of the true parameter to test
        
        """

        if (self.mark) :
            print("No theoretical garantee associated to this test")

        cov_mat = np.cov(self.params_estim[:,index_1], self.params_estim[:,index_2]) 
        stat = (self.params_estim[:,index_1] - self.params_estim[:,index_2])/ np.sqrt( cov_mat[0,0]+ cov_mat[1,1]-2*cov_mat[0,1]  )
        ks_test = kstest( stat, cdf='norm')
        
        
        with r_inline_plot():
            normality_test( robjects.FloatVector(stat))
       

        return( {"estimatorList": stat,  "KStest_stat": ks_test.statistic, "KStest_pval" : ks_test.pvalue })







            

            
