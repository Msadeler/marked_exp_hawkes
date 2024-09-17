import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats



class exp_thinning_hawkes(object):
    """
    Univariate Hawkes process with exponential kernel. No events or initial condition before initial time.
    """

    def __init__(self, m, a, b, t=0.0, max_jumps=None, max_time=None):
        """
        Parameters
        ----------
        m : float
            Baseline constant intensity.
        a : float
            Interaction factor.
        b : float
            Decay factor.
        t : float, optional
            Initial time. The default is 0.
        max_jumps : float, optional
            Maximal number of jumps. The default is None.
        max_time : float, optional
            Maximal time horizon. The default is None.
            
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
            
            
        ## Examples 
        
        """
        self.a = a
        self.b = b
        self.t_0 = t
        self.t = t
        self.m = m
        self.max_jumps = max_jumps
        self.max_time = max_time
        self.timestamps = [t]
        self.intensity_jumps = [m]
        self.aux = 0
        self.simulated = False

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

        candidate_intensity = self.m

        while flag < self.max_jumps:

            upper_intensity = max(self.m, candidate_intensity)

            self.t += np.random.exponential(1 / upper_intensity)
            
            candidate_intensity = self.m + self.aux * np.exp(-self.b * (self.t - self.timestamps[-1]))

            if upper_intensity * np.random.uniform() <= candidate_intensity:
                self.timestamps += [self.t]
                self.intensity_jumps += [candidate_intensity + self.a]
                self.aux = candidate_intensity - self.m + self.a
                flag += 1

        self.max_time = self.timestamps[-1]
        # We have to add a "self.max_time = self.timestamps[-1] at the end so plot_intensity works correctly"

    def simulate_time(self):
        """
        Simulation is done until an event that surpasses the time horizon (self.max_time) appears.
        """
        flag = self.t < self.max_time

        while flag:
            upper_intensity = max(self.m,
                                  self.m + self.aux * np.exp(-self.b * (self.t - self.timestamps[-1])))

            self.t += np.random.exponential(1 / upper_intensity)
            candidate_intensity = self.m + self.aux * np.exp(-self.b * (self.t - self.timestamps[-1]))

            flag = self.t < self.max_time

            if upper_intensity * np.random.uniform() <= candidate_intensity and flag:
                self.timestamps += [self.t]
                self.intensity_jumps += [candidate_intensity + self.a]
                self.aux = self.aux * np.exp(-self.b * (self.t - self.timestamps[-2])) + self.a

        self.timestamps+= [self.max_time]

    def plot_intensity(self, ax=None, plot_N=True):
        """
        Plot intensity function. If plot_N is True, plots also step function N([0,t]).
        The parameter ax allows to plot the intensity function in a previously created plot.

        Parameters
        ----------
        ax : .axes.Axes or array of Axes, optional.
            If None, method will generate own figure.
            Otherwise, will use given axes. Must be '.axes.Axes' if plot_N = False and array of shape (2,1) if True.
        plot_N : bool, optional.
            Whether we plot the step function N or not.
        """
        if not self.simulated:
            print("Simulate first")

        else:
            if plot_N:
                if ax is None:
                    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                elif isinstance(ax[0], matplotlib.axes.Axes):
                    ax1, ax2 = ax
                else:
                    return "ax must be a (2,1) axes"
            else:
                if ax is None:
                    fig, ax1 = plt.subplots()
                elif isinstance(ax, matplotlib.axes.Axes):
                    ax1 = ax
                else:
                    return "ax must be an instance of an axes"

            self.timestamps.append(self.max_time)

            times = np.array([self.timestamps[0], self.timestamps[1]])
            intensities = np.array([self.m, self.m])
            step = 0.01
            for i, lambda_k in enumerate(self.intensity_jumps):
                if i != 0:
                    T_k = self.timestamps[i]
                    nb_step = np.maximum(100, np.floor((self.timestamps[i + 1] - T_k) / step))
                    aux_times = np.linspace(T_k, self.timestamps[i + 1], int(nb_step))
                    times = np.append(times, aux_times)
                    intensities = np.append(intensities, self.m + (lambda_k - self.m) * np.exp(
                        -self.b * (aux_times - T_k)))

            ax1.plot([0, self.max_time], [0, 0], c='k', a=0.5)
            #if self.a < 0:
                #ax1.plot(times, intensities, label="Underlying intensity", c="#1f77b4")
            ax1.plot(times, np.maximum(intensities, 0), label="Conditional intensity", c='r')
            ax1.legend()
            ax1.grid()
            if plot_N:
                ax2.step(self.timestamps, np.append(np.arange(0, len(self.timestamps) - 1), len(self.timestamps) - 2),
                         where="post", label="$N(t)$")
                ax2.legend()
                ax2.grid()

    def set_time_intensity(self, timestamps):
    
        """
        Method to initialize a Hawkes process with a given list of timestamps. 
        
        It computes the corresponding intensity with respect to the parameters given at initialization.
        
        Parameters
        ----------
        timestamps : list of float
            Imposed jump times. Intensity is adjusted to this list of times. Must be ordered list of times.
            It is best if obtained by simulating from another instance of Hawkes process.

        """
        
        if not self.simulated:
            self.timestamps = timestamps
            self.max_time = timestamps[-1]

            intensities = [self.m]
            for k in range(1, len(timestamps)):
                intensities += [self.m + (intensities[-1] - self.m) * np.exp(
                    -self.b * (timestamps[k] - timestamps[k - 1])) + self.a]
            self.intensity_jumps = intensities
            self.simulated = True

        else:
            print("Already simulated")
            
            
    def compute_intensity(self, t):
        
        """
        Compute intensity function at time t
        """
        

        if not self.simulated:
            raise ValueError("Simulate first")
            
        if t > self.max_time:
            raise ValueError('Processus as not been simulated long enought')

        jump_before = np.max(np.where(t>= np.array(self.timestamps) )[0])
        
        T_k = self.timestamps[jump_before]
        lambda_k = self.intensity_jumps[jump_before]
       
        
        lambda_t = self.m + (lambda_k-self.m)*np.exp( -self.b*(t - T_k))
        
        return(lambda_t)

    def compensator_transform(self, plot=None, exclude_values=0):
        """
        Obtains transformed times for use of goodness-of-fit tests. 
        
        Transformation obtained through time change theorem.

        Parameters
        ----------
        plot : .axes.Axes, optional.
            If None, then it just obtains the transformed times, otherwise plot the Q-Q plot. The default is None
        exclude_values : int, optional.
            If 0 then takes all transformed points in account during plot. Otherwise, excludes first 'exclude_values'
            values from Q-Q plot. The default is 0.
        """

        if not self.simulated:
            print("Simulate first")

        else:

            T_k = self.timestamps[1]

            compensator_k = self.m * (T_k - self.t_0)

            self.timestamps_transformed = [compensator_k]
            self.intervals_transformed = [compensator_k]

            for k in range(2, len(self.timestamps)):

                lambda_k = self.intensity_jumps[k-1]
                tau_star = self.timestamps[k] - self.timestamps[k - 1]
                if lambda_k >= 0:
                    C_k = lambda_k - self.m
                else:
                    C_k = -self.m
                    tau_star -= (np.log(-(lambda_k - self.m)) - np.log(self.m)) / self.b

                compensator_k = self.m * tau_star + (C_k / self.b) * (1 - np.exp(-self.b * tau_star))

                self.timestamps_transformed += [self.timestamps_transformed[-1] + compensator_k]
                self.intervals_transformed += [compensator_k]

            if plot is not None:
                stats.probplot(self.intervals_transformed[exclude_values:], dist=stats.expon, plot=plot)


class exp_thinning_hawkes_marked(object):
    """
    Univariate Hawkes process with exponential kernel. No events or initial condition before initial time.
    

    Parameters
    ----------
    m : float
        Baseline constant intensity.
    a : float
        Interaction factor.
    b : float
        Decay factor.
    t : float, optional
        Initial time. The default is 0.
    max_jumps : float, optional
        Maximal number of jumps. The default is None.
    max_time : float, optional
        Maximal time horizon. The default is None.
        
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

    def __init__(self,
                 m,
                 a,
                 b, 
                 t=0.0, 
                 max_jumps=None, 
                 max_time=None, 
                 mark_process = False, 
                 phi = lambda x: 1, 
                 F = lambda x: 1,
                 arg_phi = {}, 
                 arg_F = {}):
        
        
        """
        Parameters
        ----------
        m : float
            Baseline constant intensity.
        a : float
            Interaction factor.
        b : float
            Decay factor.
        t : float, optional
            Initial time. The default is 0.
        max_jumps : float, optional
            Maximal number of jumps. The default is None.
        max_time : float, optional
            Maximal time horizon. The default is None.
            
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
        self.a = a
        self.b = b
        self.t_0 = t
        self.t = t
        self.m = m
        self.max_jumps = max_jumps
        self.max_time = max_time
        self.timestamps = [(t,0)]
        self.mark = 0 
        self.intensity_jumps = [m]
        self.aux = 0
        self.simulated = False
        self.F = F
        self.phi = phi
        self.arg_F = arg_F
        self.arg_phi = arg_phi
        self.mark_process = mark_process 
        
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

        candidate_intensity = self.m

        while flag < self.max_jumps:

            upper_intensity = max(self.m, candidate_intensity)

            self.t += np.random.exponential(1 / upper_intensity)
            self.mark = self.F(np.random.uniform(), **self.arg_F)
            
            candidate_intensity = self.m + self.aux * np.exp(-self.b * (self.t - self.timestamps[-1][0]))

            if upper_intensity * np.random.uniform() <= candidate_intensity:
                
                self.timestamps += [(self.t,self.mark)]
                self.intensity_jumps += [candidate_intensity + self.a*self.phi(self.timestamps[-1][1], **self.arg_phi, **self.arg_F)]
                self.aux = candidate_intensity - self.m + self.a*self.phi(self.timestamps[-1][1], **self.arg_phi, **self.arg_F)
                flag += 1

        self.max_time = self.timestamps[-1]
        
        if not self.mark_process:
            self.timestamps= [time for time, mark in self.timestamps]
            self.max_time = self.timestamps[-1]

    def simulate_time(self):
        """
        Simulation is done until an event that surpasses the time horizon (self.max_time) appears.
        """
        flag = self.t < self.max_time

        while flag:
            upper_intensity = max(self.m,
                                  self.m + self.aux * np.exp(-self.b * (self.t - self.timestamps[-1][0])))

            self.t += np.random.exponential(1 / upper_intensity)
            self.mark = self.F(np.random.uniform() , **self.arg_F) 
            candidate_intensity = self.m + self.aux * np.exp(-self.b * (self.t - self.timestamps[-1][0]))

            flag = self.t < self.max_time

            if upper_intensity * np.random.uniform() <= candidate_intensity and flag:
                self.timestamps += [(self.t,self.mark)]
                self.intensity_jumps += [candidate_intensity + self.a*self.phi(self.mark, **self.arg_phi, **self.arg_F)]
                self.aux = self.aux * np.exp(-self.b * (self.t - self.timestamps[-2][0])) + self.a*self.phi(self.timestamps[-1][1], **self.arg_phi, **self.arg_F)

        self.timestamps += [(self.max_time,0)]
        
        if not self.mark_process: 
            self.timestamps = [time for time, mark in self.timestamps]
                      
    
    def plot_intensity(self, ax=None, plot_N=True):
        """
        Plot intensity function. If plot_N is True, plots also step function N([0,t]).
        The parameter ax allows to plot the intensity function in a previously created plot.

        Parameters
        ----------
        ax : .axes.Axes or array of Axes, optional.
            If None, method will generate own figure.
            Otherwise, will use given axes. Must be '.axes.Axes' if plot_N = False and array of shape (2,1) if True.
        plot_N : bool, optional.
            Whether we plot the step function N or not.
        """
        if not self.simulated:
            print("Simulate first")

        else:
            if plot_N:
                if ax is None:
                    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                elif isinstance(ax[0], matplotlib.axes.Axes):
                    ax1, ax2 = ax
                else:
                    return "ax must be a (2,1) axes"
            else:
                if ax is None:
                    fig, ax1 = plt.subplots()
                elif isinstance(ax, matplotlib.axes.Axes):
                    ax1 = ax
                else:
                    return "ax must be an instance of an axes"

            if self.max_jumps is not None:  
                self.timestamps.append(self.timestamps[-1])
            
            if not self.mark_process:
                self.timestamps = [(time,1) for time in self.timestamps]

            times = np.array([self.timestamps[0][0], self.timestamps[1][0]])
            intensities = np.array([self.m, self.m])
            
            
            step = 0.01
            for i, lambda_k in enumerate(self.intensity_jumps):
                

                if i != 0:
                    T_k = self.timestamps[i][0]
                    nb_step = np.maximum(100, np.floor((self.timestamps[i + 1][0] - T_k) / step))
                    aux_times = np.linspace(T_k, self.timestamps[i + 1][0], int(nb_step))
                    times = np.append(times, aux_times)
                    intensities = np.append(intensities, self.m + (lambda_k - self.m) * np.exp(-self.b * (aux_times - T_k)))

            ax1.plot([0, self.timestamps[-1][0]], [0, 0], c='k', alpha=0.5)
            #if self.a < 0:
                #ax1.plot(times, intensities, label="Underlying intensity", c="#1f77b4")
            ax1.plot(times, intensities,label="Underlying intensity", c="#1f77b4", linestyle="--")
            ax1.plot(times, np.maximum(intensities, 0), label="Conditional intensity", c='r')
            
            ax1.legend()
            ax1.grid()
            if plot_N:
                ax2.step([time for time, mark in self.timestamps], np.append(np.arange(0, len(self.timestamps) - 1), len(self.timestamps) - 2),
                         where="post", label="$N(t)$")
                ax2.legend()
                ax2.grid()
                
            if not self.mark:
                self.timestamps = [time for time, mark in self.timestamps]

            if self.max_jumps is not None :            
                self.timestamps.pop()


class exp_thinning_hawkes_multi_marked(object):
    
    
    def __init__(self,
                 m,
                 a,
                 b, 
                 n=1,
                 t=0.0, 
                 mark_process = False, 
                 max_jumps=None, 
                 max_time=None, 
                 phi = lambda x: 1, 
                 F = lambda x: 1,
                 arg_phi = {}, 
                 arg_F = {}):
        
        
        """
        Parameters
        ----------
        m : float
            Baseline constant intensity.
        a : float
            Interaction factor.
        b : float
            Decay factor.
        n : int
            number of i.i.d repetition to generate
        t : float, optional
            Initial time. The default is 0.
        max_jumps : float, optional
            Maximal number of jumps. The default is None.
        max_time : float, optional
            Maximal time horizon. The default is None.
            
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
        self.a = a
        self.b = b
        self.t_0 = t
        self.t = t
        self.m = m
        self.max_jumps = max_jumps
        self.max_time = max_time
        self.timestamps = [(t,0)]
        self.mark = 0 
        self.intensity_jumps = [m]
        self.aux = 0
        self.simulated = False
        self.F = F
        self.phi = phi
        self.arg_F = arg_F
        self.arg_phi = arg_phi 
        self.nb_iter = n
        self.mark_process = mark_process
        
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


    def simulate_jumps_onces(self):
        """
        Simulation is done until the maximal number of jumps (self.max_jumps) is attained.
        """
        flag = 0

        candidate_intensity = self.m

        while flag < self.max_jumps:

            upper_intensity = max(self.m, candidate_intensity)

            self.t += np.random.exponential(1 / upper_intensity)
            self.mark = self.F(np.random.uniform() , **self.arg_F)
            
            candidate_intensity = self.m + self.aux * np.exp(-self.b * (self.t - self.timestamps[-1][0]))

            if upper_intensity * np.random.uniform() <= candidate_intensity:
                
                self.timestamps += [(self.t,self.mark)]
                self.intensity_jumps += [candidate_intensity + self.a*self.phi(self.timestamps[-1][1], **self.arg_phi, **self.arg_F)]
                self.aux = candidate_intensity - self.m + self.a*self.phi(self.timestamps[-1][1], **self.arg_phi, **self.arg_F)
                flag += 1

        self.max_time = self.timestamps[-1]

    def simulate_time_onces(self):
        """
        Simulation is done until an event that surpasses the time horizon (self.max_time) appears.
        """
        flag = self.t < self.max_time

        while flag:
            upper_intensity = max(self.m,
                                  self.m + self.aux * np.exp(-self.b * (self.t - self.timestamps[-1][0])))

            self.t += np.random.exponential(1 / upper_intensity)
            self.mark = self.F(np.random.uniform(), **self.arg_F) 
            candidate_intensity = self.m + self.aux * np.exp(-self.b * (self.t - self.timestamps[-1][0]))

            flag = self.t < self.max_time

            if upper_intensity * np.random.uniform() <= candidate_intensity and flag:
                self.timestamps += [(self.t,self.mark)]
                self.intensity_jumps += [candidate_intensity + self.a*self.phi(self.mark, **self.arg_phi, **self.arg_F)]
                self.aux = self.aux * np.exp(-self.b * (self.t - self.timestamps[-2][0])) + self.a*self.phi(self.timestamps[-1][1], **self.arg_phi, **self.arg_F)

        self.timestamps += [(self.max_time,0)]
              
        
    def simulate_time(self):
        """
        Simulation of n sample of the process
        """
        
        self.timeList = []

        for k in range(self.nb_iter):
            
            self.simulate_time_onces()
            
            if not self.mark_process:
                self.timeList+=[[time for time,mark in self.timestamps]]
            else :
                self.timeList+=[self.timestamps]
            
            self.timestamps = [self.timestamps[0]]
            self.t = self.t_0



    def simulate_jumps(self):
        """
        Simulation of n sample of the process
        """
        self.timeList = []

        for k in range(self.nb_iter):
            self.simulate_jumps_onces()
            
            if not self.mark_process:
                self.timeList+=[[time for time,mark in self.timestamps]]
            else :
                self.timeList+=[self.timestamps]
            
            self.timestamps = [self.timestamps[0]]
            self.t = self.t_0

            
            
            
            


        
        
    
  
    

