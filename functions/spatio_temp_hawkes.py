

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from  scipy.stats.contingency import crosstab
from functions.kernel_spatial_component import *

class spatiotemporal_hawkes(object):
    """
    Univariate Spatiotemporal Hawkes process on the unitary ball with exponential kernel and a PC for the baseline intensity. No events or initial condition before initial time.
    

    Parameters
    ----------
    mu_pc : float
        Baseline constant intensity at the place field.
    mu_npc : float
        Baseline constant intensity far away from the place field.
    pc : array of shape (2,)
        Coordonate of th place field
    r : float 
        Caracteristic radius of the place field
    alpha : float
        Interaction factor.
    beta : float
        Decay factor.
    t : float, optional
        Initial time. The default is 0.
    max_jumps : float, optional
        Maximal number of jumps. The default is None.
    max_time : float, optional
        Maximal time horizon. The default is None.
    kernel : 'exp' or 'indic' of function
        Indicate the kernel function to use for baseline intensity. if a function is given, it is used to compute intensity. This function should have as argument mu_pc, mu_npc, radius
        See class_and_func.kernel_spatial_component to check for the precomputed function
    Attributes
    ----------
    t_0 : float
        Initial time provided at initialization.
    timestamps : list of float
        List of simulated events. It includes the initial time t_0.
    intensity_jumps : list of float
        List of intensity at each simulated jump. It includes the baseline intensity lambda_0.
    aux : float
        Parameter used in simulation.
    simulated : bool
        Parameter that marks if a process has been already been simulated, or if its event times have been initialized.
    plot_baseline_intensity : function
        plot the baseline intensity function on the B(0,1)
    plot_baseline_realisation : function
        plot the realisation of the process on B(0,1)
    real_time_intensity: function 
        plot the intensity of the process on B(0,1) across time 
    real_time_realisation: function
        plot the realisation of the process on B(0,1) across time        
    """

    def __init__(self,
                 mu_pc, 
                 mu_npc,
                 pc,
                 radius,
                 alpha,
                 beta, 
                 t=0, 
                 x0 = np.zeros((2,)),
                 max_jumps = None, 
                 max_time = None, 
                 kernel='exp'):
        
        if mu_npc> mu_pc:
            print("Baseline intensity lower on the place field ")

        elif radius<=0 : 
            print("Radius must be stricly positive")

        else:
            self.mu_pc = mu_pc
            self.mu_npc = mu_npc
            self.pc = pc.reshape(2,1)
            self.radius = radius
            self.alpha = alpha
            self.beta = beta
            self.t = t
            self.timestamps = [t]
            self.localisation = [x0.reshape(2,1)]
            self.intensity_jump = [0]
            self.aux = 0 
            self.simulated = False
            self.max_jumps = max_jumps
            self.max_time = max_time
        

        if kernel =='exp':
            self.mu_kernel = exp_kernel
        elif kernel == 'indic':
            self.mu_kernel = indic_kernel
        elif isinstance(kernel, function):
            self.mu_kernel = kernel

        else:
            print("Unknown kernel fonction for baseline intensity, possible option are 'indic' or 'exp'")


    def simulate(self):
        """
        Auxiliary function to check if already simulated and, if not, which simulation to launch.

        Simulation follows Ogata's adapted thinning algorithm.

        Works with both self-exciting and self-regulating processes.
        
        To launch simulation either self.max_jumps or self.max_time must be other than None, so the algorithm knows when to stop.
        """
            
        if not self.simulated:
            if self.max_jumps is not None and self.max_time is None:
                self.simulate_jumps()
            elif self.max_time is not None and self.max_jumps is None:
                self.simulate_time()
            else:
                print("Either max_jumps or max_time must be given.")
            self.simulated = True

        else:
            print("Process already simulated")

    

    def simulate_jumps(self):
        """
        Simulation is done until the maximal number of jumps (self.max_jumps) is attained.
        """

        flag = 0
        
        while flag < self.max_jumps:

            upper_intensity = max(self.mu_pc, self.aux)
            
            # draw candidate time
            self.t += np.random.exponential(1 / upper_intensity)

            ## draw a random point on the B(0,1)
            r =   np.sqrt(np.random.uniform())
            theta = np.random.uniform() * 2 * np.pi
            x  = np.array([r*np.cos(theta), r*np.sin(theta)]).reshape(2,1)
           
            # compute mu(x)
            mu_x = self.mu_kernel(x-self.pc, self.mu_pc, self.mu_npc,self.radius)[0]
            #compute lambda(t,x)
            candidate_intensity = mu_x + self.aux*np.exp(-self.beta*(self.t - self.timestamps[-1]))

            ## thinning procedure 
            if upper_intensity*np.random.uniform()<= candidate_intensity:
                self.timestamps+=[self.t]
                self.localisation +=[x]

                self.aux = candidate_intensity + self.alpha- mu_x
                self.intensity_jump+=[self.aux]
                flag += 1
        
        self.max_time = self.timestamps[-1]



    def simulate_time(self):

        """
        Simulation is done until an event that surpasses the time horizon (self.max_time) appears.
        """
                
        flag = self.t < self.max_time

        while flag:
            
            upper_intensity = max(self.mu_pc, self.aux)
            
            self.t += np.random.exponential(1 / upper_intensity)

            r =   np.sqrt(np.random.uniform())
            theta = np.random.uniform() * 2 * np.pi
            x  = np.array([r*np.cos(theta), r*np.sin(theta)]).reshape(2,1)
            
            mu_x = self.mu_npc + (self.mu_pc - self.mu_npc)* np.exp( - np.linalg.norm(x-self.pc)**2/(2*self.radius))

            candidate_intensity = mu_x + self.aux*np.exp(-self.beta*(self.t - self.timestamps[-1]))

            flag = self.t < self.max_time
        
            if upper_intensity*np.random.uniform()<= candidate_intensity and flag :
                self.timestamps+=[self.t]
                self.localisation +=[x]

                self.aux = candidate_intensity + self.alpha- mu_x
                self.intensity_jump+=[self.aux]
                flag = self.t < self.max_time

    def plot_baseline_intensity(self, ax=None,bin=0.05):

        """
        Plot the baseline intensity function the square having (-1,-1), (-1,1), (1,1) and (1,-1) as vertex

        Parameters
        ----------
        ax : .axes.Axes or array of Axes, optional.
            If None, method will generate own figure.
            Otherwise, will use given axes. Must be '.axes.Axes' if plot_N = False and array of shape (2,1) if True.
        bin : positiv float, default value 0.01.
            Parameter of space discretisation

        """

        if not self.simulated :
            print("Process must be simulated first")
        else: 
            if ax is None:
                fig, ax1 = plt.subplots()
            elif isinstance(ax, matplotlib.axes.Axes):
                ax1 = ax
            else:
                return "ax must be an instance of an axes"
            
            X,Y = np.meshgrid(np.arange(-1-bin,1+bin,bin), np.arange(-1-bin,1+bin,bin))
            grid = np.array([X,Y])
            Z  = self.mu_kernel((grid.T- self.pc.T).T, self.mu_pc, self.mu_npc,self.radius)

            pc = ax1.pcolormesh(X, Y, Z)
            ax1.set_title("Baseline intensity of the process")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            fig.colorbar(pc)
            fig.show()

    def plot_realisation(self,ax=None, bin =0.05):

        """
        Plot the number of event on each bin of space

        Parameters
        ----------
        ax : .axes.Axes or array of Axes, optional.
            If None, method will generate own figure.
            Otherwise, will use given axes. Must be '.axes.Axes' if plot_N = False and array of shape (2,1) if True.
        
        bin : positiv float, default value 0.01.
            Parameter of space discretization

        """
        
        if not self.simulated :
            print("Process must be simulated first")
        else:
            if ax is None:
                fig, ax1 = plt.subplots(figsize=(15,10))
            elif isinstance(ax, matplotlib.axes.Axes):
                ax1 = ax
            else:
                return "ax must be an instance of an axes"
            
            locac = pd.DataFrame(np.concatenate(self.localisation, axis=1).T, columns = ['X', 'Y']).iloc[1:,].transform( lambda x : pd.cut(x, [k for k in np.arange(-1,1+bin,bin)],right=False))
            place_field = pd.crosstab(locac.X, locac.Y).T.sort_index(ascending=False)

            ax1.set_title("Number of event per unity of space")
            sns.heatmap(place_field, ax = ax1, cmap='coolwarm')

    def real_time_intensity(self, ax=None, bin=0.05, step = 0.5):

        """
        Plot the intensity function on the space, according to the time 

        Parameters
        ----------
        ax : .axes.Axes or array of Axes, optional.
            If None, method will generate own figure.
            Otherwise, will use given axes. Must be '.axes.Axes' if plot_N = False and array of shape (2,1) if True.
        
        bin : positiv float, default value 0.01.
            Parameter of space discretization

        step : positiv float, default value 0.1.
            Parameter of time discretization

        """
        if not self.simulated :
            print("Process must be simulated first")
        else:

            if ax is None:
                fig, ax1 = plt.subplots(figsize=(15,10))
            elif isinstance(ax, matplotlib.axes.Axes):
                ax1 = ax
            else:
                return "ax must be an instance of an axes"
       
                
            X,Y = np.meshgrid(np.arange(-1,1+bin,bin), np.arange(1,-1-bin,-bin))
            grid = np.array([X,Y])


            mu_x  = self.mu_kernel((grid.T- self.pc.T).T, self.mu_pc, self.mu_npc,self.radius)
            data = mu_x.reshape( (1,mu_x.shape[0], mu_x.shape[1]) )

            time_series = np.array([0])

            self.timestamps+= [self.max_time]

            for i in range(1, len(self.timestamps)):
                old_time = self.timestamps[i-1]
                time = self.timestamps[i]

                time_i = np.linspace(old_time, time, max(10, int((time-old_time)/step)) )

    
                time_attenuation = self.intensity_jump[i-1]*np.exp(-self.beta*(time_i-old_time))
                field_value = np.multiply.outer(time_attenuation,mu_x)


                time_series = np.concatenate((time_series, time_i), axis=0)            
                data = np.concatenate((data, field_value), axis=0)

            self.timestamps.pop()

            sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=data.min(), vmax=data.max()))
            fig.colorbar(sm)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')

            for i  in range(data.shape[0]):
                ax1.clear()
                ax1.imshow(data[i,:,:], vmin= data.min(), vmax=data.max(), cmap='coolwarm')
                ax1.set_title(time_series[i])
                plt.pause(0.001)
    
    def real_time_realisation(self,ax=None, bin=0.05, step = None):


        """
        Plot the realiation od the process on the space, according to the time 

        Parameters
        ----------
        ax : .axes.Axes or array of Axes, optional.
            If None, method will generate own figure.
            Otherwise, will use given axes. Must be '.axes.Axes' if plot_N = False and array of shape (2,1) if True.
        
        bin : positiv float, default value 0.01.
            Parameter of space discretization

        step : None or positiv float, default value None.
            Parameter of time discretization. If None, step is taken equal to (max_time - t0)/500 

        """


        if not self.simulated :
            print("Process must be simulated first")
        else:

            if ax is None:
                fig, ax1 = plt.subplots(figsize=(15,10))
            elif isinstance(ax, matplotlib.axes.Axes):
                ax1 = ax
            else:
                return "ax must be an instance of an axes"
            
            if not step:

                step = (self.max_time - self.timestamps[0])/500
    
            X,Y = np.meshgrid(np.arange(-1,1+bin,bin), np.arange(1,-1-bin,-bin))

            locac = pd.DataFrame(np.concatenate(self.localisation, axis=1).T, columns = ['X', 'Y']).iloc[1:,].transform(lambda x : pd.cut(x, [k for k in np.arange(-1,1+bin,bin)],right=False), axis=1)
            locac.insert(2,'time',self.timestamps[1:])
            time_interval = np.arange(self.timestamps[0], self.max_time+step, step)
            locac['time']= pd.cut(locac.time, time_interval )

            nb_event = np.stack(locac.groupby('time', observed=False).apply(lambda x : crosstab(x.X, x.Y, levels = (np.unique(locac.X), np.unique(locac.Y))).count))

            sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=nb_event.min(), vmax=nb_event.max()))
            fig.colorbar(sm)

            for i  in range(nb_event.shape[0]):
                ax1.clear()
                ax1.imshow(nb_event[i,:,:], vmin= nb_event.min(), vmax=nb_event.max(), cmap='coolwarm')
                ax1.set_title(time_interval[i])
                plt.pause(0.001)




        
        




        




