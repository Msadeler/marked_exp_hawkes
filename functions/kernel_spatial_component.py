import numpy as np


"""

    Function kernel for a spatial temporal Hawkes Process 

"""


def exp_kernel(x, mu_pc, mu_npc, r):
    return( mu_npc + (mu_pc-  mu_npc)*np.exp(-1/(2*r)*np.linalg.norm(x,axis=0)**2) )


def indic_kernel(x, mu_pc, mu_npc, r):
    return( mu_pc*( np.linalg.norm(x,axis=0) <= r) + mu_npc*( np.linalg.norm(x, axis=0) > r) )
