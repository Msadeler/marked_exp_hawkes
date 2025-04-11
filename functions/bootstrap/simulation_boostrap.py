import numpy as np
import scipy
import pandas as pd
from functions.bootstrap.rec_formula import *
from tqdm import tqdm



class simu_bootstrap_multidim(object):

    """
    Multivariate Hawkes process with exponential kernel. No events or initial condition before initial time.
    """

    def __init__(self,
                 times_hawkes,
                 theta,
                 s=0.0, 
                 B =1,
                 intensity_jump = None,
                 rec_compensator = rec_compensator_multidim_EHP, 
                 time_transformed = None):
        
        self.theta = theta

        if isinstance(theta, np.ndarray):
        
            self.dim = int(np.sqrt(1 + theta.shape[0]) - 1)
            
            self.m = np.array(theta[:self.dim]).reshape((self.dim, 1))
            self.a = np.array(theta[self.dim:self.dim * (self.dim + 1)]).reshape((self.dim, self.dim))
            self.b = np.array(theta[self.dim * (self.dim + 1):]).reshape((self.dim, 1))
            
        else:
            self.m, self.a, self.b = (i.copy() for i in theta)
            self.dim = len(self.m)

        self.times_hawkes = times_hawkes
        self.t_0 = s
        self.s = s
        self.B = B
        self.aux = 0
        self.simulated = False
        self.rec_compensator= rec_compensator
        self.time_transformed = np.array(time_transformed).reshape(-1, self.dim)
        self.intensity_jump= np.array(intensity_jump).reshape(-1, self.dim)
        self.rec_compensator = rec_compensator


    def simu_one_rep(self):

        """
        Auxiliary function to simulate one bootstrap repetition
        """

        self.tlistboost = [(self.t_0, 0)]

        for compo in range(self.dim):

            index = 0 
            
            self.s = self.t_0 +  np.random.exponential(1)
            
            flag = self.s < self.time_transformed[-1, compo-1]

            mcomp,acomp, bcomp = self.m[compo,0], self.a[compo,:], self.b[compo,0]

            while flag :     
                l = (self.s>self.time_transformed[index:, compo]).argmin(axis=0)
                index += l
                if index <= 1:
                    tboost = self.times_hawkes[0][0]+ self.s/ mcomp

                else : 
                    tboost = scipy.optimize.brentq( lambda t : self.rec_compensator(t, mcomp,
                                                                                    acomp[self.times_hawkes[index-1][1]-1],
                                                                                    bcomp, 
                                                                                    self.times_hawkes[index-1][0],
                                                                                    self.intensity_jump[index-2, compo],
                                                                                    self.time_transformed[index-1, compo])-self.s, 
                                                    self.times_hawkes[index-1][0],  self.times_hawkes[index][0])
                    

                self.tlistboost += [(tboost, compo+1)]
                self.s += np.random.exponential(1)
                flag = self.s < self.time_transformed[-1,compo]

        self.tlistboost = list(pd.DataFrame(self.tlistboost, columns=['time', 'comp']).sort_values('time').itertuples(index = False, name=None))
        self.tlistboost += [(self.times_hawkes[-1][0], 0)]
        
    def simulate(self):
        """
        Simulation of bootstrap repetition
        """

        self.timeList = []

        print("Start bootstrap simulation")
        for k in tqdm(range(self.B)):
            
            self.simu_one_rep()
            self.timeList+=[self.tlistboost]
            self.s = self.t_0



class simu_boostrap(object):
    """
    Univariate Hawkes process with exponential kernel. No events or initial condition before initial time.
    """

    def __init__(self,
                 times_hawkes,
                 theta,
                 arg_phi= {}, 
                 arg_f = {} ,
                 F = lambda x : 1,
                 s=0.0, 
                 B =1,
                 mark_process = False, 
                 intensity_jump = None,
                 rec_compensator = rec_compensator_unidim_MEHP, 
                 time_transformed = None,
                 phi = lambda x: 1):
        """
        Parameters
        ----------
        theta : array or list
            Parameters of the model
        t : float, optional
            Initial time. The default is 0.
        B : number of boo
        intensity_jump : list
            list_containing the value of the intensity at each time Tk
        rec_compensator : function
            Function that gives the value for a time t  in (Tk, Tk+1).
            The function must takes the following arguments : t,m,a,b,arg_f,arg_phi, phi, timebefore (the time Tk), intensity_last_jump (value of the intensity at Tk), time_transformed (the value of the compensator at Tk).
            and return the value of the compensator at the point t, for t in the segment (Tk, Tk+1).
        phi: function
            Impact function of the mark on the process
        F: function
            Cumulative distribution function of the mark
        arg_phi: dictionnary
            Argument use, other than mark, for the function F
        arg_F: dictionnary
            Argument use, other than mark and time, for the function F
            
        Attributes
        ----------
        t_0 : float
            Initial time provided at initialization.
        timestamps : list of float
            List of simulated events. It includes the initial time t_0.
        intensity_jumps : list of float
            List of intensity at each simulated jump. It includes the baseline intensity m.
        aux : float
            Parameter used in simulation.
        simulated : bool
            Parameter that marks if a process has been already been simulated, or if its event times have been initialized.
        """


        self.theta = theta
        self.times_hawkes = times_hawkes
        self.m,self.a,self.b = np.array(self.theta[-3:])
        self.t_0 = s
        self.s = s
        self.B = B
        self.aux = 0
        self.simulated = False
        self.phi = phi
        self.F = F
        self.arg_phi = arg_phi
        self.arg_f = arg_f
        self.mark_process = mark_process 
        self.rec_compensator= rec_compensator
        self.time_transformed = np.array(time_transformed)

        if intensity_jump is None: 
            print("intensity_jump must be providen")

        if not self.mark_process: 
            self.times_hawkes = [(time, 0) for time in self.times_hawkes]

        
        self.intensity_jump = intensity_jump



    def simulate_one_rep(self):
        """
        Simulation is done until the maximal number of jumps (self.max_jumps) is attained.
        """
    
        self.tlistboost = [(self.t_0, 0)]
        self.s  = self.t_0 + np.random.exponential(1)
        self.mark = self.F(np.random.uniform(), **self.arg_f)
        
        flag = self.s < self.time_transformed[-1]

        index = 0
        
        
        while flag : 
            
            l = (self.s>self.time_transformed[index:]).argmin()
            index += l
            if index <= 1:
                tboost = self.times_hawkes[0][0]+ self.s/ self.m
            else : 

                tboost = scipy.optimize.brentq( lambda t : self.rec_compensator(t, self.m,self.a,self.b,self.arg_f,self.arg_phi, self.phi, 
                                                                            self.times_hawkes[index-1], 
                                                                            self.intensity_jump[index-2],
                                                                            self.time_transformed[index-1])-self.s, 
                                            self.times_hawkes[index-1][0], self.times_hawkes[index][0])
            
            self.tlistboost+=[(tboost, self.mark)]
            self.s += np.random.exponential(1)
            breakpoint()
            self.mark = self.F(np.random.uniform(), **self.arg_f)

            flag = self.s < self.time_transformed[-1]
        
        self.tlistboost +=[self.times_hawkes[-1]]

    def simulate(self):
        """
        Simulation of bootstrap repetition
        """
        self.timeList = []

        for k in tqdm(range(self.B)):
            
            self.simulate_one_rep()
            
            self.timeList+=[self.tlistboost]
            
            self.s = self.t_0