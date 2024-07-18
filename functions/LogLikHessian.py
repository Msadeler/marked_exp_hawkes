""" 
Function to compute an approx of the fisher info for the hawkes process
"""

import numpy as np
import scipy



def approx_fisher_multi(list_time, theta, dim):


    """
        Compute en estimator of the fisher matrix for an multidimensionnal Hawkes Process without mark
        when there is a unique constant of time for each subprocess and only excitation is considered

        Output
        ----------
        Squarre matrix of shape 2*dim+dim**2, diag per block, approximating the fisher info



        Parameters
        ----------

        list_time : list of tuple of shape (n,)
            List of tuple containing the arrival times and the associated component. 
            Each tuple being of the form (T,m) with T the time of arrival and m the associated component

        theta : array of shape (d,)
            flatten array containing the values ( (mu_i), (a_ij), (b_i))

        dim : int
            number of subprocess


    """

    mu = theta[:dim]
    a = theta[dim : dim**2+dim].reshape(dim, dim)
    b = theta[dim**2+dim:]

     


    #### matrix stocking each fisher info associated to each subprocess, I_j
    #### I_j is the jth line of the matrix
    
    info_fisher = np.zeros((dim, (dim+2)**2))

     ### partial_lambda_Tk is a mtrix containing the value of partial_thetai lambda_thetai(Tk)
     ### the jth line correspond to the jth subprocess 

    partial_lambda_Tk = np.zeros((dim,dim+2))

    ### initialisation with the value in T1
     
    lambdaTk = mu 
    last_time, last_compo = list_time[1]


    partial_lambda_Tk[:,0]= 1
    info_fisher[last_compo-1, : ] += lambdaTk[last_compo-1]**(-2)* np.outer( partial_lambda_Tk[last_compo-1,:], partial_lambda_Tk[last_compo-1,:]).flatten()

    for time,compo in list_time[2:-1]:        
        delta_jump = time-last_time

        ## update the value of lambda(Tk)
        lambdaTk = mu + (lambdaTk + a[:,last_compo-1]- mu)*np.exp(-b*delta_jump)

        ## update of the partial derivate against b_i 
        partial_lambda_Tk[:,-1] = np.exp(-b*delta_jump)*partial_lambda_Tk[:,-1] - delta_jump*np.exp(delta_jump)*( a[:,last_compo-1] + np.sum(  a*(partial_lambda_Tk[:,dim-1:2*dim-1]).reshape(dim,dim) , axis =-1))


        ## update of the partial derivate against a_ij
        partial_lambda_Tk[:, dim-1:2*dim-1]= np.exp(-b*delta_jump)*( 1 + partial_lambda_Tk[:, dim-1:2*dim-1])


        info_fisher[compo-1,:] +=  lambdaTk[last_compo-1]**(-2)* np.outer( partial_lambda_Tk[last_compo-1,:], partial_lambda_Tk[last_compo-1,:]).flatten()


        last_time, last_compo = time, compo

    ## final fisher info composed a the diag block previously compued
    diag_fisher_info = scipy.linalg.block_diag(*list(map(lambda x : x.reshape(dim+2,dim+2), info_fisher)))

    return(diag_fisher_info)



def fisher_mat_approx(tList, markList, mu, a, b, f_param ,gamma=0):
    
    partial_theta_log_lik= np.array([0]*4).reshape(4,1)
    
    int_partial_lambda =  np.array([0]*4).reshape(4,1)
    
    lambda_Tk = mu
    partial1_lambaTk = 1
    partial2_lambaTk = 0
    partial3_lambaTk = 0
    partial4_lambaTk = 0

    
    last_time = tList[1]
    last_mark = markList[1]
    
    b1 = b**(-1)
    
    
    ## initialisation entre 0 et T1
    
    partial_theta_log_lik[0,0] = partial1_lambaTk/lambda_Tk 
    int_partial_lambda[0,0] = last_time
    
    
    for time, mark in zip(tList[2:-1], markList[2:-1]):
       
        
       int_partial_lambda = int_partial_lambda + np.array([ time-last_time , 
                                                           
                                                  b1*( 1- np.exp( -b*(time-last_time))) * (partial2_lambaTk+1) , 
                                                  
                                                  -b1**2* ( 2- np.exp( -b*(time-last_time))* ( b*( time-last_time )+1))*(lambda_Tk + a - mu) + b1*(( 1- np.exp( -b*(time-last_time))) )* partial3_lambaTk, 
                                                  
                                                  b1*(1- np.exp( -b*(time- last_time)))* (partial4_lambaTk + a*last_mark - a/f_param)
                                                  
                                                  ]).reshape(4,1)

       
       partial2_lambaTk = np.exp( -b*(time-last_time))*(partial2_lambaTk + 1 )
       partial3_lambaTk = -(time-last_time)* np.exp( -b*(time-last_time)) *(lambda_Tk+a - mu ) +  np.exp( -b*(time-last_time))*partial3_lambaTk
       partial4_lambaTk =  np.exp( -b*(time-last_time))*(partial4_lambaTk + a*last_mark -a/f_param )
       lambda_Tk = (lambda_Tk+ a-mu)*np.exp( - b*( time-last_time)) + mu
       
       
       partial_theta_log_lik = partial_theta_log_lik + np.array( [1,partial2_lambaTk,partial3_lambaTk, partial4_lambaTk ]).reshape(4,1)/lambda_Tk
       
       
       last_time, last_mark = time, mark


         
    time = tList[-1]
    mark = markList[-1]
    
    int_partial_lambda = int_partial_lambda + np.array([ time-last_time , 
                                               b1*( 1- np.exp( -b*(time-last_time)))* partial2_lambaTk , 
                                               
                                               -b1**2* ( 2- np.exp( -b*(time-last_time))* ( b*( time-last_time )+1))*(lambda_Tk + a - mu) + b1*(( 1- np.exp( -b*(time-last_time))) )* partial3_lambaTk, 
                                               
                                               b1*(1- np.exp( -b*(time- last_time)))* (partial4_lambaTk + a*last_mark - a/f_param)
                                               
                                               ]).reshape(4,1)
    
    
    
    
    partial_log_lik = partial_theta_log_lik - int_partial_lambda
    print(partial_theta_log_lik[-1,0])
    print(int_partial_lambda[-1,0])

    
    return(partial_log_lik)


def hessian_multivariate_different_b(lambda0, a, b, tList):
    
    
    dim = lambda0.shape[0]
    
    block_hessian = [np.zeros((2*dim+1,2*dim+1)) for k in range(dim)]
    
    
    
    
    
    jump_compo = np.array(list(map( lambda x: x[1], tList)))
    jump_time = np.array(list(map( lambda x: x[0], tList)))
    
    previous_time =  jump_time[1]
    
    diff_time = np.array([])
    
    a_jump = a[ :,tList[1][1]-1].reshape((dim,1))
    b_jump = b[:,tList[1][1]-1].reshape((dim,1))


    block_hessian[jump_compo[1]-1][0,0] = -1/(lambda0[jump_compo[1]-1,0]**(2))

    
    for k in range(len(tList[2:-1])):
        
        time, compo = jump_time[k+2], jump_compo[k+1]
        
        diff_time = np.array([time-previous_time] + (time -previous_time +  diff_time ).tolist())

        
        
                    
        lambda_i_Tk = lambda0[compo-1,0] + np.sum( a_jump[compo-1,:]*np.exp(- b_jump[compo-1,:]*diff_time ))
        
        
        partial_aij_Tk = np.zeros((dim))
        partial_aij_bij = np.zeros((dim))
        partial_bij_b_ij = np.zeros((dim))
        
        
        int_partial_bij_b_ij = [np.zeros((dim)) for k in range(dim)]
        int_partial_aij_b_ij = [np.zeros((dim)) for k in range(dim)]

        
        for j in range(1,dim+1):
            
            
            
            jump_j_compo = np.where(jump_compo[:k+2]==j)[0]
            
            
            ## Correspond la la liste (T_(k)  - T^j_l) avec T_(k) temps de saut auquel on est
            ## et (T^j_l) temps de saut de la j-ème composante qui sont strictement plus petits que T_(k)
            
            diff_j_time = diff_time[ (jump_j_compo-1).tolist()]
            
            
            
            ## Calcul des dérivés partielles de l'intensité de la composante qui saute, à l'intant de saut
 
            partial_aij_Tk[j-1] = np.sum( np.exp( -b[compo-1, j-1]* diff_j_time))
            partial_aij_bij[j-1] = - np.sum(  diff_j_time*np.exp( -b[compo-1, j-1]*diff_j_time ) )
            partial_bij_b_ij[j-1] = a[compo-1,j-1]*np.sum(  (diff_j_time**2) *np.exp( -b[compo-1, j-1]*diff_j_time ))
            
     
            

            ## Calcul de l'integrale de chaque intensité entre previous_time et time  
        
            int_partial_bij_b_ij[j-1] = a[j-1,:]/(b[j-1,:]**(3))* np.array( list( map( lambda b : 
                                                                                                   np.sum(np.exp( -b*(previous_time - time + diff_j_time))* ( (b*(previous_time - time + diff_j_time )+1)**2 +1 ))  - 
                                                                                                   np.sum( np.exp(- b*diff_j_time)* (( b*( diff_j_time)+1)**2 + 1)), b[j-1,:] )))
            int_partial_aij_b_ij[j-1] = -(1/b[j-1,:]**(2))* np.array( list( map( lambda x : 
                                                                                          np.sum(( (x*(previous_time - time + diff_j_time )+1 ))*np.exp( -x*(previous_time - time + diff_j_time)))  - 
                                                                                          np.sum( ( x*( diff_j_time)+1)*np.exp(- x*diff_j_time)), b[j-1,:] )))
            
            
            block_hessian[j-1][ (dim+1):, (dim+1):] -= np.diag( int_partial_bij_b_ij[j-1])
            block_hessian[j-1][ 1:(dim+1), (dim+1):] -= np.diag( int_partial_aij_b_ij[j-1])

            
        partial_bij_Tk = a[compo-1, :]* partial_aij_bij
        
        

        
        block_hessian[compo-1][0,0]  -= 1/(lambda_i_Tk**(2))
        block_hessian[compo-1][0, 1:(dim+1)] -= partial_aij_Tk/(lambda_i_Tk**(2) )
        block_hessian[compo-1][0,(dim+1):] -=   partial_bij_Tk/(lambda_i_Tk**(2))
        
        
        block_hessian[compo-1][1:(dim+1), 1:(dim+1)] -= 1/(lambda_i_Tk**(2))*np.outer(partial_aij_Tk,partial_aij_Tk)
        
        block_hessian[compo-1][1:(dim+1), (dim+1):] +=  1/lambda_i_Tk * np.diag(partial_aij_bij) -1/(lambda_i_Tk**2)*np.outer(partial_aij_Tk,partial_bij_Tk  )
        
        block_hessian[compo-1][ (dim+1):, (dim+1):] +=  1/(lambda_i_Tk) *np.diag(partial_bij_b_ij) - 1/(lambda_i_Tk**(2))*np.outer(partial_bij_Tk,partial_bij_Tk)
        
            
        a_jump = np.concatenate((a[:,compo-1].reshape(dim,1),a_jump), axis=1)
        b_jump = np.concatenate( (b[:,compo-1].reshape(dim,1),b_jump), axis=1)
        previous_time = time 

   
 
    time = tList[-1][0]
    diff_time = np.array([time-previous_time] + (time -previous_time +  diff_time ).tolist())

    
    int_partial_bij_b_ij = [np.zeros((dim)) for k in range(dim)]
    int_partial_aij_b_ij = [np.zeros((dim)) for k in range(dim)]
            
    
    
    
    for j in range(1,dim+1):
        jump_j_compo = np.where(jump_compo[:-1]==j)[0]
        
        
        ## Correspond la la liste (T_(k)  - T^j_l) avec T_(k) temps de saut auquel on est
        ## et (T^j_l) temps de saut de la j-ème composante qui sont plus petits que T_(k)
        
        diff_j_time = diff_time[ (jump_j_compo-1).tolist()]
    
         
        int_partial_bij_b_ij[j-1] = a[j-1,:]/(b[j-1,:]**(3))* np.array( list( map( lambda b : np.sum(np.exp( -b*(previous_time - time + diff_j_time))* ( (b*(previous_time - time + diff_j_time )+1)**2 +1 ))  - np.sum( np.exp(- b*diff_j_time)* (( b*( diff_j_time)+1)**2+1)), b[j-1,:] )))
        int_partial_aij_b_ij[j-1] = -1/(b[j-1,:]**(2))* np.array( list( map( lambda b : np.sum(( (b*(previous_time - time + diff_j_time )+1 ))* np.exp( -b*(previous_time - time + diff_j_time)))  - np.sum( ( b*( diff_j_time)+1)*np.exp(- b*diff_j_time)), b[j-1,:] )))
        
        
        block_hessian[j-1][(dim+1):, (dim+1):] =block_hessian[compo-1][(dim+1):, (dim+1):] - np.diag(int_partial_bij_b_ij[j-1])
        
        block_hessian[j-1][1:(dim+1):, (dim+1):] = block_hessian[j-1][1:(dim+1):, (dim+1):] - np.diag(int_partial_aij_b_ij[j-1])


        ### On symetrise chaque sous matrice
        
        
    for j in range(dim):
        
        block_hessian[j-1][1:(dim+1), 0 ] =  block_hessian[j-1][0,1:(dim+1) ] 
        block_hessian[j-1][(dim+1):, 0 ] =  block_hessian[j-1][0,dim+1: ] 

        block_hessian[j-1][(dim+1):, 1:(dim+1)] = block_hessian[j-1][ 1:(dim+1), (dim+1):]
        
    
    
    return(-1/time * scipy.linalg.block_diag(*block_hessian))


def likelihood_hessien_unidim(m, a, b, tList, Tmax):
    
        nbJump = len(tList)

        diff_time = list(map( lambda i : tList[i]- tList[:i],  [k for k in range(1,nbJump)])) 
        
        
        lambda_Tk2 =  np.array([m] + list(map( lambda x :   m + a*np.sum(np.exp(-b*x)), diff_time )))
        
        partial2_Tk2 =np.array( [0] + list(map(  lambda x : np.sum(np.exp(-b*x)), diff_time)))
        
        partial23_Tk2 = np.array( [0] + list(map(  lambda x : -np.sum(x*np.exp(-b*x)), diff_time)))
        partial3_Tk2 = a*partial23_Tk2
        
        partial33_Tk = np.array( [0] + list(map(  lambda x : a*np.sum((x**2)*np.exp(-b*x)), diff_time))) 
        
        int_partial_23_Tk2 = -b**(-2)* np.array( [0]  + [ 1 - np.exp(-b*diff_time[0][0])*(b*(diff_time[0][0])+1) ] + 
                                             list(map(lambda i : 1 +
                                                          np.sum( (b*diff_time[i-1]+1)*np.exp(-b*diff_time[i-1])) - 
                                                          np.sum( (b*diff_time[i]+1)*np.exp(-b*diff_time[i])),
                                                      [k for k in range(1,len(diff_time))])))
        
        int_partial_33_Tk2 = a*b**(-3)* np.array( [0]  + [ 2 - np.exp(-b*diff_time[0][0])*(  (b*(diff_time[0][0])+1)**2 + 1 ) ] + 
                                             list(map(lambda i : 2 +
                                                          np.sum( ( (b*diff_time[i-1]+1)**2 +1  )*np.exp(-b*diff_time[i-1])) - 
                                                          np.sum( ((b*diff_time[i]+1)**2 +1)  *np.exp(-b*diff_time[i])),
                                                      [k for k in range(1,len(diff_time))])))
        
         
        lambda_Tk_minus_1 =  lambda_Tk2**(-1)
        lambda_Tk_minus_2 =  lambda_Tk2**(-2)


        partial11_log = - np.sum( lambda_Tk_minus_2) 
        
        partial12_log = -np.sum( partial2_Tk2* lambda_Tk_minus_2)
        
        patial_13_log =- np.sum( partial3_Tk2*lambda_Tk_minus_2 )
        
        partial22_log =  -np.sum( lambda_Tk_minus_2*partial2_Tk2**2 )
        
        partial23_log =  np.sum( partial23_Tk2* lambda_Tk_minus_1 - partial2_Tk2*partial3_Tk2*lambda_Tk_minus_2 -  int_partial_23_Tk2)
        
        partial33_log = np.sum(   partial33_Tk*lambda_Tk_minus_1 - lambda_Tk_minus_2*partial3_Tk2**2 -int_partial_33_Tk2 )
        
        
        diff_tmax = Tmax - tList
        
        int_partial_33_T2 = a*b**(-3)* (2 + 
                                              np.sum( np.exp(-b*diff_time[-1])*(  (b*diff_time[-1] +1)**2 +1 )) - 
                                              np.sum( np.exp(-b*diff_tmax)*(  (b*diff_tmax +1)**2 +1 )) ) 

        
        
        int_partial_23_T2 = -b**(-2)*( 1 +
                                         np.sum( (b*diff_time[-1]+1)*np.exp(-b*diff_time[-1])) - 
                                         np.sum( (b*diff_tmax+1)*np.exp(-b*diff_tmax)) 
                                                         )
        
        hessian2 = np.array([ [partial11_log, partial12_log, patial_13_log], 
                             [partial12_log,partial22_log, partial23_log-int_partial_23_T2 ],
                             [patial_13_log, partial23_log-int_partial_23_T2, partial33_log-int_partial_33_T2]])
     
        return( -1/Tmax*hessian2)
            
def likelihood_hessien_unidim_marked_process_with_exp_mark(lambda0, a, b, gamma,tList, mark_list, Tmax,T0=0):
    
        """
            /!\ Hessian for linear exponential hawkes process with eponential mark 
        """


        nbJump = len(tList)



        ## Liste des (T_k - T_j) pour j plus petit que k strictement et k supérieur à 2
        
        diff_time = list(map( lambda i :   tList[i]- tList[:i],  [k for k in range(1,nbJump)])) 
        
        
        ## Vecteur contenant les valeurs de la fonction lambda évaluée en T_k

        lambda_Tk =  np.array([lambda0] + list(map( lambda k:   lambda0 + a*np.sum(np.exp(-b*diff_time[k]+ gamma*mark_list[:1+k]) ),[k for k in range(len(diff_time))])))
        
        
        
        partial2_Tk =np.array( [0] + list(map(  lambda k : np.sum(np.exp(-b*diff_time[k] + gamma*mark_list[:1+k])), 
                                               [k for k in range(len(diff_time))])))
        
        partial24_Tk = np.array( [0] + list( map( lambda k : np.sum(np.exp(-b*diff_time[k] + gamma*mark_list[:1+k])*mark_list[:1+k]) , 
                                                [k for k in range(len(diff_time))])))
        
        
        partial4_Tk = a* partial24_Tk
        
        partial23_Tk = np.array( [0] + list(map(  lambda k : -np.sum(diff_time[k] *np.exp(-b*diff_time[k] + gamma* mark_list[:1+k])), 
                                                [k for k in range(len(diff_time))])))
        
        
        partial3_Tk = a*partial23_Tk
        
        partial33_Tk = np.array( [0] + list(map(  lambda k : a*np.sum((diff_time[k]**2)*np.exp(-b*diff_time[k]+ gamma*mark_list[:1+k] )), 
                                                [k for k in range(len(diff_time))]))) 
        
        partial34_Tk = - a* np.array( [0] + list(map(  lambda k : np.sum((diff_time[k]*mark_list[:1+k])*np.exp(-b*diff_time[k]+ gamma*mark_list[:1+k] )), 
                                                [k for k in range(len(diff_time))]))) 
        
        partial44_Tk = a* np.array( [0] + list(map(  lambda k : np.sum( (mark_list[:1+k]**2)*np.exp(-b*diff_time[k]+ gamma*mark_list[:1+k] )), 
                                                [k for k in range(len(diff_time))])))
        
        
        
        
        
        int_partial_23_Tk = -b**(-2)* np.array( [0]  + [ np.exp( gamma*mark_list[0])*(1- np.exp( -b*diff_time[0][0] )*( b*diff_time[0][0] +1) ) ] + 
                                             list(map(lambda i : np.exp(gamma*mark_list[i]) +
                                                          np.sum( (b*diff_time[i-1]+1)*np.exp(-b*diff_time[i-1] + gamma*mark_list[:i])  ) - 
                                                          np.sum( (b*diff_time[i]+1)*np.exp(-b*diff_time[i] + gamma*mark_list[:i+1] )) ,
                                                      [k for k in range(1,len(diff_time))])))
        
        
        int_partial_24_Tk = b**(-1)* np.array( [0] +  [ np.exp( gamma*mark_list[0]) *(1- np.exp( -b*diff_time[0][0]) ) ] +
                                                 list(map( lambda k:  mark_list[k]*np.exp( gamma*mark_list[k])+ 
                                                                 np.sum(  mark_list[:k]*np.exp( gamma*mark_list[k] -b*diff_time[k-1])) -
                                                                 np.sum(  mark_list[:k+1]*np.exp( gamma*mark_list[:k+1] -b*diff_time[k])), 
                                                                 [k for k in range(1,len(diff_time))])))
        
        int_partial_33_Tk = a*b**(-3)* np.array( [0]  + [ np.exp( gamma*mark_list[0])*(2 - np.exp(-b*diff_time[0][0])*(  (b*(diff_time[0][0])+1)**2 + 1 )) ] + 
                                             list(map(lambda i : 2*np.exp( gamma*mark_list[i])  +
                                                          np.sum( ( (b*diff_time[i-1]+1)**2 +1  )*np.exp(-b*diff_time[i-1] + gamma*mark_list[:i] )) - 
                                                          np.sum( ((b*diff_time[i]+1)**2 +1)  *np.exp(-b*diff_time[i] +  gamma*mark_list[:i+1])),
                                                      [k for k in range(1,len(diff_time))])))
        
         
        int_partial_34_Tk = -a* np.array( [0]  + [ mark_list[1]*np.exp( gamma*mark_list[1])*(1- np.exp( -b*diff_time[0][0] )*( b*diff_time[0][0] +1) ) ] + 
                                             list(map(lambda i : np.exp(gamma*mark_list[1+i]) +
                                                          np.sum( mark_list[:i]*(b*diff_time[i-1]+1)*np.exp(-b*diff_time[i-1] + gamma*mark_list[:i])  ) - 
                                                          np.sum( mark_list[:i+1]*(b*diff_time[i]+1)*np.exp(-b*diff_time[i] + gamma*mark_list[:i+1] )) ,
                                                      [k for k in range(1,len(diff_time))])))
        
        
        int_partial_44_Tk = a*b**(-1) * np.array( [0] +  [ (mark_list[0]**2) * np.exp( gamma*mark_list[0]) *(1- np.exp( -b*diff_time[0][0]) ) ] +
                                                 list(map( lambda k:  (mark_list[k]**2)* np.exp( gamma*mark_list[k])+ 
                                                                 np.sum(  (mark_list[:k]**2) * np.exp( gamma*mark_list[:k] -b*diff_time[k-1])) -
                                                                 np.sum( (mark_list[:k+1]**2) * np.exp( gamma*mark_list[:k+1] -b*diff_time[k])), 
                                                                 [k for k in range(1,len(diff_time))])))
        
        
        lambda_Tk_minus_1 =  lambda_Tk**(-1)
        lambda_Tk_minus_2 =  lambda_Tk**(-2)


        partial11_log = - np.sum( lambda_Tk_minus_2) 
        
        partial12_log = -np.sum( partial2_Tk* lambda_Tk_minus_2)
        
        partial13_log =- np.sum( partial3_Tk*lambda_Tk_minus_2)
        
        partial14_log = - np.sum( partial4_Tk*lambda_Tk_minus_2 )
        
        partial22_log =  -np.sum( lambda_Tk_minus_2*partial2_Tk**2 )
        
        partial23_log =  np.sum( partial23_Tk* lambda_Tk_minus_1 - partial2_Tk*partial3_Tk*lambda_Tk_minus_2-  int_partial_23_Tk)
       
        partial24_log =  np.sum( partial24_Tk* lambda_Tk_minus_1 - partial2_Tk*partial4_Tk*lambda_Tk_minus_2 -  int_partial_24_Tk)

        partial34_log = np.sum(partial34_Tk *lambda_Tk_minus_1 - partial3_Tk*partial4_Tk*lambda_Tk_minus_2 - int_partial_34_Tk )
        
        
        
        partial33_log = np.sum(   partial33_Tk*lambda_Tk_minus_1 - lambda_Tk_minus_2*partial3_Tk**2 -int_partial_33_Tk )
        
        partial44_log = np.sum(   partial44_Tk*lambda_Tk_minus_1 - lambda_Tk_minus_2*partial4_Tk**2 -int_partial_44_Tk )

        
        
        
        
        diff_tmax = Tmax - tList
        
        int_partial_33_T = a*b**(-3)* (2*np.exp( gamma* mark_list[-1])  + 
                                              np.sum( np.exp(-b*diff_time[-1] + gamma*mark_list[:-1])*(  (b*diff_time[-1] +1)**2 +1 )) - 
                                              np.sum( np.exp(-b*diff_tmax+  gamma*mark_list)*(  (b*diff_tmax +1)**2 +1 )) ) 

        
        
        int_partial_23_T = -b**(-2)*( np.exp( gamma* mark_list[-1]) +
                                         np.sum( (b*diff_time[-1]+1)*np.exp(-b*diff_time[-1] +  gamma*mark_list[:-1] )) - 
                                         np.sum( (b*diff_tmax+1)*np.exp(-b*diff_tmax +  gamma*mark_list)) 
                                                          )
     
        int_partial_24_T = b**(-1)*( np.exp( gamma* mark_list[-1])*mark_list[-1] +
                                         np.sum( mark_list[:-1]*np.exp(-b*diff_time[-1] +  gamma*mark_list[:-1] )) - 
                                         np.sum( mark_list*np.exp(-b*diff_tmax +  gamma*mark_list)) 
                                                          )
        
        int_partial_34_T = -a*( np.exp( gamma* mark_list[-1])*mark_list[-1] +
                                         np.sum( mark_list[:-1]*(b*diff_time[-1]+1)*np.exp(-b*diff_time[-1] +  gamma*mark_list[:-1] )) - 
                                         np.sum(mark_list*(b*diff_tmax+1)*np.exp(-b*diff_tmax +  gamma*mark_list)) 
                                                          )
     
     
        int_partial_44_T = a*b**(-1)*( np.exp( gamma* mark_list[-1])*mark_list[-1]**2 +
                                         np.sum(( mark_list[:-1]**2 )*np.exp(-b*diff_time[-1] +  gamma*mark_list[:-1] )) - 
                                         np.sum( (mark_list**2)*np.exp(-b*diff_tmax +  gamma*mark_list)) 
                                                          )
        
     
        
        
        
        hessian = np.array([ [partial11_log, partial12_log, partial13_log, partial14_log ], 
                            
                             [partial12_log,partial22_log, partial23_log-int_partial_23_T, partial24_log - int_partial_24_T ],
                             
                             [partial13_log, partial23_log-int_partial_23_T, partial33_log-int_partial_33_T, partial34_log - int_partial_34_T], 
                             
                             [partial14_log, partial24_log- int_partial_24_T, partial34_log- int_partial_34_T, partial44_log- int_partial_44_T ]])
     
        return( -hessian/Tmax )
  

def likelihood_hessien_marked_process(lambda0, a, b,gamma, phi,phi_deriv, phi_deriv_deriv, tList, mark_list, Tmax):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    
     ### /!\  Hessian for linear exponential hawkes process with an impact function \phi  that is C1
     ### phi_deriv is the first derivate of phi and phi_deriv_deriv is it's second one
    
        nbJump = len(tList)



        ## Liste des (T_k - T_j) pour j plus petit que k strictement et k supérieur à 2
        
        diff_time = list(map( lambda i :   tList[i]- tList[:i],  [k for k in range(1,nbJump)])) 
        
        
        ## Vecteur contenant les valeurs de la fonction lambda évaluée en T_k

        lambda_Tk =  np.array([lambda0] + list(map( lambda k:   lambda0 + a*np.sum(np.exp(-b*diff_time[k])*phi(mark_list[:1+k],gamma)),[k for k in range(len(diff_time))])))
        
        
        
        partial2_Tk =np.array( [0] + list(map(  lambda k : np.sum(np.exp(-b*diff_time[k])*phi(mark_list[:1+k],gamma)), 
                                               [k for k in range(len(diff_time))])))
        
        partial24_Tk = np.array( [0] + list( map( lambda k : np.sum(np.exp(-b*diff_time[k])*phi_deriv(mark_list[:1+k], gamma)) , 
                                                [k for k in range(len(diff_time))])))
        
        
        partial4_Tk = a* partial24_Tk
        
        partial23_Tk = np.array( [0] + list(map(  lambda k : -np.sum(diff_time[k] *np.exp(-b*diff_time[k])*phi(mark_list[:1+k],gamma)), 
                                                [k for k in range(len(diff_time))])))
        
        
        partial3_Tk = a*partial23_Tk
        
        partial33_Tk = np.array( [0] + list(map(  lambda k : a*np.sum((diff_time[k]**2)*np.exp(-b*diff_time[k])*phi(mark_list[:1+k],gamma)), 
                                                [k for k in range(len(diff_time))]))) 
        
        partial34_Tk = - a* np.array( [0] + list(map(  lambda k : np.sum(diff_time[k]*np.exp(-b*diff_time[k])*phi_deriv(mark_list[:1+k],gamma)), 
                                                [k for k in range(len(diff_time))]))) 
        
        partial44_Tk = a* np.array( [0] + list(map(  lambda k : np.sum( np.exp(-b*diff_time[k])* phi_deriv_deriv(mark_list[:1+k],gamma)), 
                                                [k for k in range(len(diff_time))])))
        
        
        
        
        
        int_partial_23_Tk = -b**(-2)* np.array( [0]  + [phi( mark_list[0],gamma)*(1- np.exp( -b*diff_time[0][0] )*( b*diff_time[0][0] +1) ) ] + 
                                             list(map(lambda i :phi( mark_list[i],gamma) +
                                                          np.sum( (b*diff_time[i-1]+1)*np.exp(-b*diff_time[i-1])*phi( mark_list[:i],gamma)  ) - 
                                                          np.sum( (b*diff_time[i]+1)*np.exp(-b*diff_time[i])*phi( mark_list[:i+1],gamma) )  ,
                                                      [k for k in range(1,len(diff_time))])))
        
        
        int_partial_24_Tk = b**(-1)* np.array( [0] +  [ phi_deriv( mark_list[0],gamma) *(1- np.exp( -b*diff_time[0][0]) ) ] +
                                                 list(map( lambda k:  phi_deriv( mark_list[k],gamma)+ 
                                                                 np.sum(  phi_deriv( mark_list[:k],gamma)*np.exp( -b*diff_time[k-1])) -
                                                                 np.sum(  phi_deriv( mark_list[:k+1],gamma)*np.exp(-b*diff_time[k])), 
                                                                 [k for k in range(1,len(diff_time))])))
        
        
        
        int_partial_33_Tk = a*b**(-3)* np.array( [0]  + [ phi( mark_list[0],gamma)*(2 - np.exp(-b*diff_time[0][0])*(  (b*(diff_time[0][0])+1)**2 + 1 )) ] + 
                                             list(map(lambda i : 2*np.exp( gamma*mark_list[i])  +
                                                          np.sum( ( (b*diff_time[i-1]+1)**2 +1  )*np.exp(-b*diff_time[i-1])*phi(mark_list[:i],gamma)) - 
                                                          np.sum( ((b*diff_time[i]+1)**2 +1)  *np.exp(-b*diff_time[i])*phi(mark_list[:i+1],gamma)),
                                                      [k for k in range(1,len(diff_time))])))
        
         
        int_partial_34_Tk = -a* np.array( [0]  + [ phi_deriv( mark_list[0],gamma)*(1- np.exp( -b*diff_time[0][0] )*( b*diff_time[0][0] +1) ) ] + 
                                             list(map(lambda i : np.exp(gamma*mark_list[1+i]) +
                                                          np.sum( phi_deriv(mark_list[:i], gamma)*(b*diff_time[i-1]+1)*np.exp(-b*diff_time[i-1])  ) - 
                                                          np.sum( phi_deriv(mark_list[:i+1], gamma)*(b*diff_time[i]+1)*np.exp(-b*diff_time[i])) ,
                                                      [k for k in range(1,len(diff_time))])))
        
        
        int_partial_44_Tk = a*b**(-1) * np.array( [0] +  [ phi_deriv_deriv(mark_list[0],gamma)*(1- np.exp( -b*diff_time[0][0]) ) ] +
                                                 list(map( lambda k:  phi_deriv_deriv(mark_list[k],gamma)+ 
                                                                 np.sum(  phi_deriv_deriv(mark_list[:k],gamma) * np.exp(  -b*diff_time[k-1])) -
                                                                 np.sum(  phi_deriv_deriv(mark_list[:k+1],gamma)* np.exp(-b*diff_time[k])), 
                                                                 [k for k in range(1,len(diff_time))])))
        
        
        lambda_Tk_minus_1 =  lambda_Tk**(-1)
        lambda_Tk_minus_2 =  lambda_Tk**(-2)


        partial11_log = - np.sum( lambda_Tk_minus_2) 
        
        partial12_log = -np.sum( partial2_Tk* lambda_Tk_minus_2)
        
        partial13_log =- np.sum( partial3_Tk*lambda_Tk_minus_2)
        
        partial14_log = - np.sum( partial4_Tk*lambda_Tk_minus_2 )
        
        partial22_log =  -np.sum( lambda_Tk_minus_2*partial2_Tk**2 )
        
        partial23_log =  np.sum( partial23_Tk* lambda_Tk_minus_1 - partial2_Tk*partial3_Tk*lambda_Tk_minus_2-  int_partial_23_Tk)
       
        partial24_log =  np.sum( partial24_Tk* lambda_Tk_minus_1 - partial2_Tk*partial4_Tk*lambda_Tk_minus_2 -  int_partial_24_Tk)

        partial34_log = np.sum(partial34_Tk *lambda_Tk_minus_1 - partial3_Tk*partial4_Tk*lambda_Tk_minus_2 - int_partial_34_Tk )
        
        
        
        partial33_log = np.sum(   partial33_Tk*lambda_Tk_minus_1 - lambda_Tk_minus_2*partial3_Tk**2 -int_partial_33_Tk )
        
        partial44_log = np.sum(   partial44_Tk*lambda_Tk_minus_1 - lambda_Tk_minus_2*partial4_Tk**2 -int_partial_44_Tk )

        
        
        
        
        diff_tmax = Tmax - tList
        
        int_partial_33_T = a*b**(-3)* (2*phi( mark_list[-1],gamma)  + 
                                              np.sum(  phi(mark_list[:-1], gamma)*np.exp(-b*diff_time[-1])*(  (b*diff_time[-1] +1)**2 +1 )) - 
                                              np.sum( phi(mark_list, gamma)*np.exp(-b*diff_tmax)*(  (b*diff_tmax +1)**2 +1 )) ) 

        
        
        int_partial_23_T = -b**(-2)*( phi( mark_list[-1],gamma)  +
                                         np.sum( phi(mark_list[:-1], gamma)*(b*diff_time[-1]+1)*np.exp(-b*diff_time[-1])) - 
                                         np.sum( phi(mark_list, gamma)*(b*diff_tmax+1)*np.exp(-b*diff_tmax)) 
                                                          )
     
        int_partial_24_T = b**(-1)*( phi_deriv( mark_list[-1],gamma) +
                                         np.sum( phi_deriv( mark_list[:-1],gamma)*np.exp(-b*diff_time[-1])) - 
                                         np.sum(  phi_deriv( mark_list,gamma)*np.exp(-b*diff_tmax )) 
                                                          )
        
        int_partial_34_T = -a*(  phi_deriv( mark_list[-1],gamma) +
                                         np.sum(phi_deriv( mark_list[:-1],gamma)*(b*diff_time[-1]+1)*np.exp(-b*diff_time[-1] )) - 
                                         np.sum(phi_deriv( mark_list,gamma)*(b*diff_tmax+1)*np.exp(-b*diff_tmax)) 
                                                          )
     
     
        int_partial_44_T = a*b**(-1)*( phi_deriv_deriv( mark_list[-1],gamma) +
                                         np.sum(phi_deriv( mark_list[:-1],gamma)*np.exp(-b*diff_time[-1] )) - 
                                         np.sum( phi_deriv( mark_list,gamma)*np.exp(-b*diff_tmax )) 
                                                          )
        
     
        
        
        
        hessian = np.array([ [partial11_log, partial12_log, partial13_log, partial14_log ], 
                            
                             [partial12_log,partial22_log, partial23_log-int_partial_23_T, partial24_log - int_partial_24_T ],
                             
                             [partial13_log, partial23_log-int_partial_23_T, partial33_log-int_partial_33_T, partial34_log - int_partial_34_T], 
                             
                             [partial14_log, partial24_log- int_partial_24_T, partial34_log- int_partial_34_T, partial44_log- int_partial_44_T ]])
     
        return( -hessian/Tmax )
  

def approx_fisher_with_stationnarity_assumption(list_time, list_mark, mu, a, b, gamma, psi):

    """
        Compute en estimator of the fisher matrix for an unidimensionnal Hawkes Process with an exponential mark


        Parameters
        ----------

        list_time : list of array of shape (n,)
            list of times beging at T0 and ending at Tmax of observation

        list_mark : list of array of shape (n,)
            list of mark, with same length as list_time, with the mark associated to each jump. the mark at T0 and Tmax do not impact the present computation

        mu : float
            excitation rate of the process 

        a : float
            interaction term of the process

        b : float 
            parameter calibrating the time of interaction 

        gamma: float 
            parameter defining the impact of the mark

        psi: float 
            parameter defining the density of marks

    """

    ## values of lambda(Tk), initialised for T1
    lambda_Tk = mu

    ## values of partial derivates of lambda at Tk, initialised for T1
    partial_lambda_Tk = np.array([1, 0, 0, 0, 0])
    aprox_fisher = np.outer(partial_lambda_Tk,partial_lambda_Tk)/lambda_Tk**2

    last_time, last_mark = list_time[1], list_mark[1]

    ## computation bu recurrence of the two precedent quantities, strarting in T2

    for time, mark in zip(list_time[2:-1], list_mark[2:-1]): 

        delta_jump = time - last_time
        time_impact = np.exp( - b*delta_jump)

        partial_mu =1 
        partial_a_Tk = time_impact*( partial_lambda_Tk[1] + (psi-gamma)/psi *np.exp(gamma*last_mark))
        partial_b_Tk = - a*partial_a_Tk*delta_jump*time_impact + time_impact*partial_lambda_Tk[2]
        partial_gamma_Tk = time_impact*( partial_lambda_Tk[3] + a*np.exp(gamma*last_mark)*(last_mark*(psi-gamma)/psi- 1/psi) )
        partial_psi_gamma_Tk = time_impact*( partial_lambda_Tk[4]+ a*gamma/psi**2*np.exp(gamma*last_mark) )

        lambda_Tk = mu + ( lambda_Tk + a*np.exp(gamma*last_mark)*(psi-gamma)/psi - mu  )*time_impact

        list_partial_lambda = np.array([ partial_mu, partial_a_Tk, partial_b_Tk, partial_gamma_Tk, partial_psi_gamma_Tk])
        aprox_fisher+= np.outer(list_partial_lambda,list_partial_lambda)/ lambda_Tk**2

        last_time, last_mark =time, mark 

    return(aprox_fisher)

