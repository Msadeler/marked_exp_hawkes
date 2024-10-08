import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import warnings


def fill_square_matrix(K):
    res = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            val = input("%s %s" % (i + 1, j + 1))
            res[i, j] = val
    return res


def fill_column_vector(K):
    res = np.empty((K, 1))
    for i in range(K):
        val = input("%s 1" % (i + 1))
        res[i, 0] = val
    return res


class multivariate_exponential_hawkes_marked(object):
    """
    Multivariate Hawkes process with exponential kernel.
    No events nor initial conditions considered.
    Mark don't depend on the component that jump and are iid
    """

    def __init__(self, 
                 m, 
                 a, 
                 b,
                 phi = lambda x: 1, 
                 F = lambda x: 1,
                 arg_phi = {}, 
                 arg_F = {},
                 max_jumps=None, 
                 max_time=None):
        """

        Parameters
        ----------
        m : array_like
            Baseline intensity vector. m.shape[0] must coincide with shapes for a and b.
        a : array_like
            Interaction factors matrix. Must be a square array with a.shape[0] coinciding with mu and b.
        b : array_like
            Decay factor matrix. Must be either an array. When corresponding to decay for each process i, it must
            be of shape (number_of_process, 1), or a square array. b.shape[0] must coincide with mu and a.
        max_jumps : float, optional
            Maximal number of jumps. The default is None.
        max_time : float, optional
            Maximal time horizon. The default is None.

        Attributes
        ----------
        nb_processes : int
            Number of dimensions.
        timestamps : list of tuple (float, int)
            List of simulated events and their marks.
        intensity_jumps : array of float
            Array containing all intensities at each jump. It includes the baseline intensities mu.
        simulated : bool
            Parameter that marks if a process has been already been simulated,
            or if its event times have been initialized.

        """


       
        self.m = m.reshape((a.shape[0], 1))
        self.a = a
        self.b = b
        self.max_jumps = max_jumps
        self.max_time = max_time
        self.F = F
        self.phi = phi
        self.arg_F  =arg_F
        self.arg_phi= arg_phi

        self.nb_processes = self.m.shape[0]
        self.count = np.zeros(self.nb_processes, dtype=int)

        self.timestamps = [(0.0, 0, 0)]
        self.intensity_jumps = np.copy(m)

        self.simulated = False

    def simulate(self):
        """
        Auxiliary function to check if already simulated and, if not, which simulation to launch.

        Simulation follows Ogata's adapted thinning algorithm. Upper bound obtained by the positive-part process.

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
            
    def simulate_time(self):
     """
     Simulation is done for a window [0, T] (T = self.max_time) is attained.
     """
     t = 0
     flag = t < self.max_time

     auxiliary_a = np.where(self.a > 0, self.a, 0)
     auxiliary_ij = np.zeros((self.nb_processes, self.nb_processes))
     auxiliary_intensity = np.copy(self.m)

     ij_intensity = np.zeros((self.nb_processes, self.nb_processes))

     while flag:

         upper_intensity = np.sum(auxiliary_intensity)

         previous_t = t
         t += np.random.exponential(1 / upper_intensity)
         mark = self.F(np.random.uniform(), **self.arg_F)
         
         ij_intensity = np.multiply(ij_intensity, np.exp(-self.b * (t - previous_t)))
         
         
         candidate_intensities = self.m + np.sum(ij_intensity, axis=1, keepdims=True)
         
         pos_candidate = np.maximum(candidate_intensities, 0) / upper_intensity
         type_event = np.random.multinomial(1,
                                            np.concatenate((pos_candidate.squeeze(), np.array([0.0])))).argmax()
         flag = t < self.max_time
         
         if type_event < self.nb_processes and flag:
             
             self.timestamps += [(t, type_event + 1, mark)]
             ij_intensity[:, type_event] += self.a[:, type_event]*self.phi(mark, **self.arg_phi, **self.arg_F)
             self.intensity_jumps = np.c_[
                 self.intensity_jumps, self.m + np.sum(ij_intensity, axis=1, keepdims=True)]

             auxiliary_ij = np.multiply(auxiliary_ij, np.exp(-self.b * (t - self.timestamps[-2][0])))
             auxiliary_ij[:, type_event] += auxiliary_a[:, type_event]*self.phi(mark, **self.arg_phi,**self.arg_F)
             auxiliary_intensity = self.m + np.sum(auxiliary_ij, axis=1, keepdims=True)

             self.count[type_event] += 1

     self.timestamps += [(self.max_time, 0, 0)]
     self.simulated = True


    def simulate_jumps(self):
        """
        Simulation is done until the maximal number of jumps (self.max_jumps) is attained.
        """
        flag = 0
        t = 0

        auxiliary_a = np.where(self.a > 0, self.a, 0)
        auxiliary_ij = np.zeros((self.nb_processes, self.nb_processes))
        auxiliary_intensity = np.copy(self.m)

        ij_intensity = np.zeros((self.nb_processes, self.nb_processes))

        while flag < self.max_jumps:
            upper_intensity = np.sum(auxiliary_intensity)

            previous_t = t
            t += np.random.exponential(1 / upper_intensity)
            mark = self.F(np.random.uniform(), **self.arg_F)


            # ij_intensity = np.multiply(ij_intensity, np.exp(-self.b * (t - self.timestamps[-1][0])))
            
            ij_intensity = np.multiply(ij_intensity, np.exp(-self.b * (t - previous_t)))
            
            candidate_intensities = self.m + np.sum(ij_intensity, axis=1, keepdims=True)
            pos_candidate = np.maximum(candidate_intensities, 0) / upper_intensity
            type_event = np.random.multinomial(1, np.concatenate((pos_candidate.squeeze(), np.array([0.0])))).argmax()
            
            if type_event < self.nb_processes:
                
                self.timestamps += [(t, type_event + 1, mark)]
                
                ij_intensity[:, type_event] += np.multiply(self.a[:, type_event],self.phi(mark, **self.arg_phi,**self.arg_F))
                
                self.intensity_jumps = np.c_[
                    self.intensity_jumps, self.m + np.sum(ij_intensity, axis=1, keepdims=True)]

                auxiliary_ij = np.multiply(auxiliary_ij, np.exp(-self.b * (t - self.timestamps[-2][0])))
                auxiliary_ij[:, type_event] += np.multiply(auxiliary_a[:, type_event], self.phi(mark, **self.arg_phi, **self.arg_F))
                auxiliary_intensity = self.m + np.sum(auxiliary_ij, axis=1, keepdims=True)

                flag += 1

                self.count[type_event] += 1

        self.max_time = self.timestamps[-1][0]
        # Important to add the max_time for plotting and being consistent.
        self.timestamps += [(self.max_time, 0)]
        self.simulated = True



        
    def plot_intensity(self, ax=None, plot_N=True):
        """
        Plot intensity function. If plot_N is True, plots also step functions N^i([0,t]).
        The parameter ax allows to plot the intensity function in a previously created plot.

        Parameters
        ----------
        ax : array of Axes, optional.
            If None, method will generate own figure.
            Otherwise, will use given axes. Must be array of shape (2,K) if plot_N = True, or (K,) if plot_N = False
        plot_N : bool, optional.
            Whether we plot the step function N^i or not.
        """

        if not self.simulated:
            print("Simulate first")

        else:
            plt.rcParams['axes.grid'] = True
            if plot_N:
                jumps_plot = [[0] for i in range(self.nb_processes)]
                if ax is None:
                    fig, ax = plt.subplots(2, self.nb_processes, sharex=True)
                    ax1 = ax[0, :]
                    ax2 = ax[1, :]
                elif isinstance(ax[0,0], matplotlib.axes.Axes):
                    ax1 = ax[0, :]
                    ax2 = ax[1, :]
                else:
                    return "ax is the wrong shape. It should be (2, number of processes+1)"
            else:
                if ax is None:
                    fig, ax1 = plt.subplots(1, self.nb_processes)
                elif isinstance(ax, matplotlib.axes.Axes) or isinstance(ax, np.ndarray):
                    ax1 = ax
                else:
                    return "ax is the wrong shape. It should be (number of processes+1,)"

            times = [0, self.timestamps[1][0]]
            intensities = np.array([[self.m[i, 0], self.m[i, 0]] for i in range(self.nb_processes)])

            ij_intensity = np.zeros((self.nb_processes, self.nb_processes))

            step = 100
            # print("here", self.timestamps[len(self.timestamps)])

            for i in range(1, len(self.timestamps[1:])):
                # On commence par mettre à jour la matrice lambda^{ij}
                ij_intensity = np.multiply(ij_intensity,
                                           np.exp(-self.b * (self.timestamps[i][0] - self.timestamps[i - 1][0])))
                # On enregistre le saut d'intensité de l'évenement, pour son type.
                ij_intensity[:, self.timestamps[i][1]-1] += self.a[:, self.timestamps[i][1]-1]

                # On définit la fonction à tracer entre T_n et T_{n+1}
                func = lambda x: self.m + np.matmul(
                    np.multiply(ij_intensity, np.exp(-self.b * (x - self.timestamps[i][0]))),
                                np.ones((self.nb_processes, 1)))

                # On enregistre la division de temps et les sauts
                interval_t = np.linspace(self.timestamps[i][0], self.timestamps[i + 1][0], step)
                times += interval_t.tolist()

                intensities = np.concatenate((intensities, np.array(list(map(func, interval_t))).squeeze().T ), axis=1)
                if plot_N:
                    jumps_plot[self.timestamps[i][1]-1] += [self.timestamps[i][0] for t in range(2)]

            for i in range(self.nb_processes):
                ax1[i].plot(times, intensities[i], label="Underlying intensity", c="#1f77b4", linestyle="--")
                ax1[i].plot(times, np.maximum(intensities[i], 0), label="Conditional intensity", c='r')
                # ax1[i].plot([i for i,j in self.timestamps[:-1]], self.intensity_jumps[i,:], c='k', a=0.5)

            ax1[0].legend()

            if plot_N:
                for i in range(self.nb_processes):
                    jumps_plot[i] += [self.max_time]
                    ax2[i].plot(jumps_plot[i], [t for t in range(self.count[i]+1) for j in range(2)], c="r", label="Process #%s"%(i+1))
                    # ax2[i].set_ylim(ax2[i].get_ylim())
                    for j in range(self.nb_processes):
                        if j != i:
                            ax2[j].plot(jumps_plot[i], [t for t in range(self.count[i]+1) for j in range(2)], c="#1f77b4", alpha=0.5)

                    ax2[i].legend()


class multivariate_exponential_hawkes(object):
    """
    Multivariate Hawkes process with exponential kernel. No events nor initial conditions considered.
    """

    def __init__(self, m, a, b, max_jumps=None, max_time=None):
        """

        Parameters
        ----------
        mu : array_like
            Baseline intensity vector. mu.shape[0] must coincide with shapes for a and b.
        a : array_like
            Interaction factors matrix. Must be a square array with a.shape[0] coinciding with mu and b.
        b : array_like
            Decay factor matrix. Must be either an array. When corresponding to decay for each process i, it must
            be of shape (number_of_process, 1), or a square array. b.shape[0] must coincide with mu and a.
        max_jumps : float, optional
            Maximal number of jumps. The default is None.
        max_time : float, optional
            Maximal time horizon. The default is None.

        Attributes
        ----------
        nb_processes : int
            Number of dimensions.
        timestamps : list of tuple (float, int)
            List of simulated events and their marks.
        intensity_jumps : array of float
            Array containing all intensities at each jump. It includes the baseline intensities mu.
        simulated : bool
            Parameter that marks if a process has been already been simulated,
            or if its event times have been initialized.

        """

        # We must begin by verifying that the process is a point process. In other words, that the number of
        # points in any bounded interval is a.s. finite. For this, we have to verify that the spectral radius of
        # the matrix a/b (term by term) is <1.

        b_radius = np.copy(b)
        b_radius[b_radius == 0] = 1
        spectral_radius = np.max(np.abs(np.linalg.eig(np.abs(a) / b_radius)[0]))

        if spectral_radius >= 1:
            # raise ValueError("Spectral radius is %s, which makes the process unstable." % (spectral_radius))
            warnings.warn("Spectral radius is %s, which makes the process unstable." % (spectral_radius),RuntimeWarning)
        self.m = m.reshape((a.shape[0], 1))
        self.a = a
        self.b = b
        self.max_jumps = max_jumps
        self.max_time = max_time

        self.nb_processes = self.m.shape[0]
        self.count = np.zeros(self.nb_processes, dtype=int)

        self.timestamps = [(0.0, 0)]
        self.intensity_jumps = np.copy(m)

        self.simulated = False

    def simulate(self):
        """
        Auxiliary function to check if already simulated and, if not, which simulation to launch.

        Simulation follows Ogata's adapted thinning algorithm. Upper bound obtained by the positive-part process.

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

    def simulate_time(self):
     """
     Simulation is done for a window [0, T] (T = self.max_time) is attained.
     """
     t = 0
     flag = t < self.max_time

     auxiliary_a = np.where(self.a > 0, self.a, 0)
     auxiliary_ij = np.zeros((self.nb_processes, self.nb_processes))
     auxiliary_intensity = np.copy(self.m)

     ij_intensity = np.zeros((self.nb_processes, self.nb_processes))

     while flag:

         upper_intensity = np.sum(auxiliary_intensity)

         previous_t = t
         t += np.random.exponential(1 / upper_intensity)
         
         ij_intensity = np.multiply(ij_intensity, np.exp(-self.b * (t - previous_t)))
         
         
         candidate_intensities = self.m + np.sum(ij_intensity, axis=1, keepdims=True)
         
         pos_candidate = np.maximum(candidate_intensities, 0) / upper_intensity
         type_event = np.random.multinomial(1,
                                            np.concatenate((pos_candidate.squeeze(), np.array([0.0])))).argmax()
         flag = t < self.max_time
         
         if type_event < self.nb_processes and flag:
             
             self.timestamps += [(t, type_event + 1)]
             ij_intensity[:, type_event] += self.a[:, type_event]
             self.intensity_jumps = np.c_[
                 self.intensity_jumps, self.m + np.sum(ij_intensity, axis=1, keepdims=True)]

             auxiliary_ij = np.multiply(auxiliary_ij, np.exp(-self.b * (t - self.timestamps[-2][0])))
             auxiliary_ij[:, type_event] += auxiliary_a[:, type_event]
             auxiliary_intensity = self.m + np.sum(auxiliary_ij, axis=1, keepdims=True)

             self.count[type_event] += 1

     self.timestamps += [(self.max_time, 0)]
     self.simulated = True

    def simulate_jumps(self):
        """
        Simulation is done until the maximal number of jumps (self.max_jumps) is attained.
        """
        flag = 0
        t = 0

        auxiliary_a = np.where(self.a > 0, self.a, 0)
        auxiliary_ij = np.zeros((self.nb_processes, self.nb_processes))
        auxiliary_intensity = np.copy(self.m)

        ij_intensity = np.zeros((self.nb_processes, self.nb_processes))

        while flag < self.max_jumps:
            upper_intensity = np.sum(auxiliary_intensity)

            previous_t = t
            t += np.random.exponential(1 / upper_intensity)

            # ij_intensity = np.multiply(ij_intensity, np.exp(-self.b * (t - self.timestamps[-1][0])))
            ij_intensity = np.multiply(ij_intensity, np.exp(-self.b * (t - previous_t)))
            candidate_intensities = self.m + np.sum(ij_intensity, axis=1, keepdims=True)
            pos_candidate = np.maximum(candidate_intensities, 0) / upper_intensity
            type_event = np.random.multinomial(1, np.concatenate((pos_candidate.squeeze(), np.array([0.0])))).argmax()
            if type_event < self.nb_processes:
                self.timestamps += [(t, type_event + 1)]
                ij_intensity[:, type_event] += self.a[:, type_event]
                self.intensity_jumps = np.c_[
                    self.intensity_jumps, self.m + np.sum(ij_intensity, axis=1, keepdims=True)]

                auxiliary_ij = np.multiply(auxiliary_ij, np.exp(-self.b * (t - self.timestamps[-2][0])))
                auxiliary_ij[:, type_event] += auxiliary_a[:, type_event]
                auxiliary_intensity = self.m + np.sum(auxiliary_ij, axis=1, keepdims=True)

                flag += 1

                self.count[type_event] += 1

        self.max_time = self.timestamps[-1][0]
        # Important to add the max_time for plotting and being consistent.
        self.timestamps += [(self.max_time, 0)]

    def plot_intensity(self, ax=None, plot_N=True):
        """
        Plot intensity function. If plot_N is True, plots also step functions N^i([0,t]).
        The parameter ax allows to plot the intensity function in a previously created plot.

        Parameters
        ----------
        ax : array of Axes, optional.
            If None, method will generate own figure.
            Otherwise, will use given axes. Must be array of shape (2,K) if plot_N = True, or (K,) if plot_N = False
        plot_N : bool, optional.
            Whether we plot the step function N^i or not.
        """

        if not self.simulated:
            print("Simulate first")

        else:
            plt.rcParams['axes.grid'] = True
            if plot_N:
                jumps_plot = [[0] for i in range(self.nb_processes)]
                if ax is None:
                    fig, ax = plt.subplots(2, self.nb_processes, sharex=True)
                    ax1 = ax[0, :]
                    ax2 = ax[1, :]
                elif isinstance(ax[0,0], matplotlib.axes.Axes):
                    ax1 = ax[0, :]
                    ax2 = ax[1, :]
                else:
                    return "ax is the wrong shape. It should be (2, number of processes+1)"
            else:
                if ax is None:
                    fig, ax1 = plt.subplots(1, self.nb_processes)
                elif isinstance(ax, matplotlib.axes.Axes) or isinstance(ax, np.ndarray):
                    ax1 = ax
                else:
                    return "ax is the wrong shape. It should be (number of processes+1,)"

            times = [0, self.timestamps[1][0]]
            intensities = np.array([[self.m[i, 0], self.m[i, 0]] for i in range(self.nb_processes)])

            ij_intensity = np.zeros((self.nb_processes, self.nb_processes))

            step = 100
            # print("here", self.timestamps[len(self.timestamps)])

            for i in range(1, len(self.timestamps[1:])):
                # On commence par mettre à jour la matrice lambda^{ij}
                ij_intensity = np.multiply(ij_intensity,
                                           np.exp(-self.b * (self.timestamps[i][0] - self.timestamps[i - 1][0])))
                # On enregistre le saut d'intensité de l'évenement, pour son type.
                ij_intensity[:, self.timestamps[i][1]-1] += self.a[:, self.timestamps[i][1]-1]

                # On définit la fonction à tracer entre T_n et T_{n+1}
                func = lambda x: self.m + np.matmul(
                    np.multiply(ij_intensity, np.exp(-self.b * (x - self.timestamps[i][0]))),
                                np.ones((self.nb_processes, 1)))

                # On enregistre la division de temps et les sauts
                interval_t = np.linspace(self.timestamps[i][0], self.timestamps[i + 1][0], step)
                times += interval_t.tolist()

                intensities = np.concatenate((intensities, np.array(list(map(func, interval_t))).squeeze().T ), axis=1)
                if plot_N:
                    jumps_plot[self.timestamps[i][1]-1] += [self.timestamps[i][0] for t in range(2)]

            for i in range(self.nb_processes):
                ax1[i].plot(times, intensities[i], label="Underlying intensity", c="#1f77b4", linestyle="--")
                ax1[i].plot(times, np.maximum(intensities[i], 0), label="Conditional intensity", c='r')
                # ax1[i].plot([i for i,j in self.timestamps[:-1]], self.intensity_jumps[i,:], c='k', a=0.5)

            ax1[0].legend()

            if plot_N:
                for i in range(self.nb_processes):
                    jumps_plot[i] += [self.max_time]
                    ax2[i].plot(jumps_plot[i], [t for t in range(self.count[i]+1) for j in range(2)], c="r", label="Process #%s"%(i+1))
                    # ax2[i].set_ylim(ax2[i].get_ylim())
                    for j in range(self.nb_processes):
                        if j != i:
                            ax2[j].plot(jumps_plot[i], [t for t in range(self.count[i]+1) for j in range(2)], c="#1f77b4", a=0.5)

                    ax2[i].legend()

    def plot_heatmap(self, ax=None):
        """
        This function allows to observe the heatmap where each cell {ij} corresponds to the value {a/b} from that interaction

        Parameters
        ----------
        ax : .axes.Axes, optional.
            If None, method will generate own ax.
            Otherwise, will use given ax.
        """
        import seaborn as sns

        if ax is None:
            fig, ax = plt.subplots()
        else:
            ax = ax
        b_heat = np.copy(self.b)
        b_heat[b_heat == 0] = 1
        heat_matrix = self.a/b_heat

        hex_list = ['#FF3333', '#FFFFFF', '#33FF49']

        ax = sns.heatmap(heat_matrix, cmap=get_continuous_cmap(hex_list), center=0, ax=ax, annot=True)





class multivariate_exponential_hawkes_marked_multi(object):
    """
    Multivariate Hawkes process with exponential kernel.
    No events nor initial conditions considered.
    Mark don't depend on the component that jump and are iid
    """

    def __init__(self, 
                 m, 
                 a, 
                 b,
                 n = 1,
                 t_0=0,
                 mark_process = False,
                 phi = lambda x: 1, 
                 F = lambda x: 1,
                 arg_phi = {}, 
                 arg_F = {},
                 max_jumps=None, 
                 max_time=None):
        """

        Parameters
        ----------
        m : array_like
            Baseline intensity vector. m.shape[0] must coincide with shapes for a and b.
        a : array_like
            Interaction factors matrix. Must be a square array with a.shape[0] coinciding with mu and b.
        b : array_like
            Decay factor matrix. Must be either an array. When corresponding to decay for each process i, it must
            be of shape (number_of_process, 1), or a square array. b.shape[0] must coincide with mu and a.
        max_jumps : float, optional
            Maximal number of jumps. The default is None.
        max_time : float, optional
            Maximal time horizon. The default is None.

        Attributes
        ----------
        nb_processes : int
            Number of dimensions.
        timestamps : list of tuple (float, int)
            List of simulated events and their marks.
        intensity_jumps : array of float
            Array containing all intensities at each jump. It includes the baseline intensities mu.
        simulated : bool
            Parameter that marks if a process has been already been simulated,
            or if its event times have been initialized.

        """


       
        self.m = m.reshape((a.shape[0], 1))
        self.a = a
        self.b = b
        self.max_jumps = max_jumps
        self.max_time = max_time
        self.F = F
        self.phi = phi
        self.arg_F  =arg_F
        self.arg_phi= arg_phi
        self.mark_process = mark_process
        self.nb_processes = self.m.shape[0]
        self.count = np.zeros(self.nb_processes, dtype=int)
        self.t_0 = t_0
        self.timestamps = [(0.0, 0, 0)]
        self.intensity_jumps = np.copy(m)
        self.nb_iter = n
        self.simulated = False

    def simulate(self):
        """
        Auxiliary function to check if already simulated and, if not, which simulation to launch.

        Simulation follows Ogata's adapted thinning algorithm. Upper bound obtained by the positive-part process.

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
            
    def simulate_time_onces(self):
     """
     Simulation is done for a window [0, T] (T = self.max_time) is attained.
     """
     t = self.t_0
     flag = t < self.max_time

     auxiliary_a = np.where(self.a > 0, self.a, 0)
     auxiliary_ij = np.zeros((self.nb_processes, self.nb_processes))
     auxiliary_intensity = np.copy(self.m)

     ij_intensity = np.zeros((self.nb_processes, self.nb_processes))

     while flag:

         upper_intensity = np.sum(auxiliary_intensity)

         previous_t = t
         t += np.random.exponential(1 / upper_intensity)
         mark = self.F(np.random.uniform() , **self.arg_F)
         
         ij_intensity = np.multiply(ij_intensity, np.exp(-self.b * (t - previous_t)))
         
         
         candidate_intensities = self.m + np.sum(ij_intensity, axis=1, keepdims=True)
         
         pos_candidate = np.maximum(candidate_intensities, 0) / upper_intensity
         type_event = np.random.multinomial(1,
                                            np.concatenate((pos_candidate.squeeze(), np.array([0.0])))).argmax()
         flag = t < self.max_time
         
         if type_event < self.nb_processes and flag:
             
             self.timestamps += [(t, type_event + 1, mark)]
             ij_intensity[:, type_event] += self.a[:, type_event]*self.phi(mark, **self.arg_phi, **self.arg_F)
             self.intensity_jumps = np.c_[
                 self.intensity_jumps, self.m + np.sum(ij_intensity, axis=1, keepdims=True)]

             auxiliary_ij = np.multiply(auxiliary_ij, np.exp(-self.b * (t - self.timestamps[-2][0])))
             auxiliary_ij[:, type_event] += auxiliary_a[:, type_event]*self.phi(mark, **self.arg_phi,**self.arg_F)
             auxiliary_intensity = self.m + np.sum(auxiliary_ij, axis=1, keepdims=True)

             self.count[type_event] += 1

     self.timestamps += [(self.max_time, 0, 0)]
     self.simulated = True
     


    def simulate_jumps_onces(self):
        """
        Simulation is done until the maximal number of jumps (self.max_jumps) is attained.
        """
        flag = 0
        t = self.t_0


        auxiliary_a = np.where(self.a > 0, self.a, 0)
        auxiliary_ij = np.zeros((self.nb_processes, self.nb_processes))
        auxiliary_intensity = np.copy(self.m)

        ij_intensity = np.zeros((self.nb_processes, self.nb_processes))

        while flag < self.max_jumps:
            upper_intensity = np.sum(auxiliary_intensity)

            previous_t = t
            t += np.random.exponential(1 / upper_intensity)
            mark = self.F(np.random.uniform(), **self.arg_F)


            # ij_intensity = np.multiply(ij_intensity, np.exp(-self.b * (t - self.timestamps[-1][0])))
            
            ij_intensity = np.multiply(ij_intensity, np.exp(-self.b * (t - previous_t)))
            
            candidate_intensities = self.m + np.sum(ij_intensity, axis=1, keepdims=True)
            pos_candidate = np.maximum(candidate_intensities, 0) / upper_intensity
            type_event = np.random.multinomial(1, np.concatenate((pos_candidate.squeeze(), np.array([0.0])))).argmax()
            
            if type_event < self.nb_processes:
                
                self.timestamps += [(t, type_event + 1, mark)]
                
                ij_intensity[:, type_event] += np.multiply(self.a[:, type_event],self.phi(mark, **self.arg_phi,**self.arg_F))
                
                self.intensity_jumps = np.c_[
                    self.intensity_jumps, self.m + np.sum(ij_intensity, axis=1, keepdims=True)]

                auxiliary_ij = np.multiply(auxiliary_ij, np.exp(-self.b * (t - self.timestamps[-2][0])))
                auxiliary_ij[:, type_event] += np.multiply(auxiliary_a[:, type_event], self.phi(mark, **self.arg_phi, **self.arg_F))
                auxiliary_intensity = self.m + np.sum(auxiliary_ij, axis=1, keepdims=True)

                flag += 1

                self.count[type_event] += 1

        self.max_time = self.timestamps[-1][0]
        # Important to add the max_time for plotting and being consistent.
        self.timestamps += [(self.max_time, 0,0)]
        self.simulated = True


    def simulate_time(self):
        """
        Simulation of n sample of the process
        """
        
        self.timeList = []

        for k in range(self.nb_iter):
            self.simulate_time_onces()

            if not self.mark_process:
                self.timeList+=[(time, dim) for time,dim,mark in self.timestamps]
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
            print(self.timestamps)
            
            if not self.mark_process:
                self.timeList+=[(time, dim) for time,dim,mark in self.timestamps]
            else :
                self.timeList+=[self.timestamps]
            
            self.timestamps = [self.timestamps[0]]
            self.t = self.t_0
