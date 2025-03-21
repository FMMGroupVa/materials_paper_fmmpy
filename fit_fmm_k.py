# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 09:46:50 2024

@author: Christian
"""

import numpy as np
from numpy.fft import fft, ifft
from scipy.optimize import minimize, Bounds
from numba import jit
import matplotlib.pyplot as plt
from auxiliar_functions import seq_times, szego, mobius, predict, predict2, transition_matrix, split_complex, inner_products_sum_2
from sklearn.linear_model import LinearRegression



def fit_fmm_k(analytic_data_matrix, time_points=None, n_back=None, max_iter=None,
              omega_grid=None, weights=None, post_optimize=True, omega_min=0.01, omega_max=1):
    
    if(analytic_data_matrix.ndim == 2):
        n_ch, n_obs = analytic_data_matrix.shape
    elif(analytic_data_matrix.ndim == 1):
        n_obs = analytic_data_matrix.shape[0]
        n_ch = 1
        analytic_data_matrix = analytic_data_matrix[np.newaxis, :]
    else:
        print("Bad data matrix dimensions.")
    
    if(max_iter==None):
        max_iter=1
        
    # Grid definition.
    fmm_grid = np.meshgrid(omega_grid, time_points)
    afd_grid = (1-fmm_grid[0])/(1+fmm_grid[0])*np.exp(1j*(fmm_grid[1]))
    
    modules_grid = (1-omega_grid)/(1+omega_grid)*np.exp(1j*0)
    an_search_len = modules_grid.shape[0]
    
    # base: DFT coefficients of szego kernels with different a's (different modules)
    base = np.zeros((modules_grid.shape[0], n_obs), dtype=complex)
    for i in range(an_search_len):
        base[i,:] = fft(szego(modules_grid[i], time_points), n_obs)
    
    # Parameters (AFD)
    coefs = np.zeros((n_ch, n_back+1), dtype=complex)
    phis = np.zeros((n_ch, n_back+1), dtype=complex)
    a_parameters = np.zeros(n_back+1, dtype=complex)

    # Start decomposing: c0 (mean)
    # Remainder R_(k+1) = ( R_k - c_k*e_ak(t) ) / m_ak(t)
    # e_ak := szego(ak, t)
    # m_ak := mobius(ak, t)
    z = np.exp(1j*time_points)
    remainder = np.copy(analytic_data_matrix)
    for ch_i in range(n_ch):
        coefs[ch_i,0] = np.mean(analytic_data_matrix[ch_i,:])
        remainder[ch_i,:] = ((analytic_data_matrix[ch_i,:] - coefs[ch_i,0])/z)
    # transition_mat = np.zeros((n_back+1, n_back+1), dtype = "complex")
    
    
    ## 1 Iteration of the backfitting algorithm: fit k waves
    for k in range(1, n_back+1):
        ## STEP 1: Grid search - AFD-FFT formulations
        abs_coefs = 0
        for ch_i in range(n_ch):
            #abs_coefs += np.abs(ifft(pymat.repmat(fft(analytic_data_matrix[ch_i, :], n_obs), an_search_len, 1) * base, n_obs, 1))
            abs_coefs += weights[ch_i]*np.abs(ifft(np.repeat(fft(
                remainder[ch_i, :], n_obs)[np.newaxis, :], 
                an_search_len, axis=0) * base, n_obs, 1))
        abs_coefs = abs_coefs.T # n_obs x n_omegas
        
        # Best a
        max_loc_tmp = np.argwhere(abs_coefs == np.amax(abs_coefs))
        best_a = afd_grid[max_loc_tmp[0, 0], max_loc_tmp[0, 1]]
        
        ## STEP 2: Postoptimization - Profile log-likelihood.
        if(post_optimize):
            res = minimize(
                inner_products_sum_2, x0=split_complex(best_a), 
                args=(remainder, time_points, weights), 
                # Bounds: (-2pi, 4pi) para explorar bien parametro circular
                method='L-BFGS-B', bounds=[(-2*np.pi, 4*np.pi), ((1-omega_max)/(1+omega_max),(1-omega_min)/(1+omega_min))],
                tol=1e-4, options={'disp': False})
            opt_a = res.x[1]*np.exp(1j*res.x[0])
            a_parameters[k] = opt_a
        else:
            a_parameters[k] = best_a
        
        # Coefficient calculations 
        szego_a = szego(a_parameters[k], time_points)
        for ch_i in range(n_ch):
            coefs[ch_i, k] = np.conj(szego_a.dot(remainder[ch_i,:].conj().T))/n_obs
            remainder[ch_i,:] = ((remainder[ch_i,:] - coefs[ch_i,k]*szego_a) 
                                 / mobius(a_parameters[k], time_points))
        
        AFD2FMM_matrix = transition_matrix(a_parameters[0:(k+1)])
        for ch_i in range(n_ch):    
            phis[ch_i, 0:k+1] = np.dot(AFD2FMM_matrix, coefs[ch_i, 0:k+1].T).T
            
    
    if max_iter > 1:
        for iter_j in range(1,max_iter):
            # Auxiliar Blaschke product: z*Bl_{a_1,...,a_K} = z*m(a1,t)*...*m(aK,t)
            blaschke = z 
            for k in range(1, n_back+1):
                blaschke = blaschke*mobius(a_parameters[k], time_points)
        
            for k in range(1, n_back+1):
                # Calculate the standard reminder (data-prediction) without component k:  r = X - sum ci*Bi, i != j
                # std_remainder = analytic_data_matrix - predict(np.delete(a_parameters, k, axis=0), np.delete(coefs, k, axis=1), time_points)
                std_remainder = analytic_data_matrix - predict2(np.delete(a_parameters, k, axis=0), analytic_data_matrix, time_points)[0]

                weights = 1/np.var(std_remainder.real, axis=1, ddof=1)
                
                # Calculate the reduced reminder reminder/(z*mob1*...,mobK) (without k)
                blaschke = blaschke / mobius(a_parameters[k], time_points)
                remainder = std_remainder / blaschke
                
                abs_coefs = 0
                for ch_i in range(n_ch):
                    #abs_coefs += np.abs(ifft(pymat.repmat(fft(analytic_data_matrix[ch_i, :], n_obs), an_search_len, 1) * base, n_obs, 1))
                    abs_coefs += weights[ch_i]*np.abs(ifft(np.repeat(fft(
                        remainder[ch_i, :], n_obs)[np.newaxis, :], 
                        an_search_len, axis=0) * base, n_obs, 1))
                    
                abs_coefs = abs_coefs.T # n_obs x n_omegas
                
                # Best a
                max_loc_tmp = np.argwhere(abs_coefs == np.amax(abs_coefs))
                best_a = afd_grid[max_loc_tmp[0, 0], max_loc_tmp[0, 1]]
                if(post_optimize):
                    res = minimize(
                        inner_products_sum_2, x0=split_complex(best_a), 
                        args=(remainder, time_points, weights), 
                        # Bounds: (-2pi, 4pi) para explorar bien parametro circular
                        method='L-BFGS-B', bounds=[(-2*np.pi, 4*np.pi), ((1-omega_max)/(1+omega_max),(1-omega_min)/(1+omega_min))],
                        options={'disp': False})
                    opt_a = res.x[1]*np.exp(1j*res.x[0])
                    a_parameters[k] = opt_a
                else:
                    a_parameters[k] = best_a
                for ch_i in range(n_ch):
                    coefs[ch_i, k] = np.conj(szego_a.dot(remainder[ch_i,:].conj().T))/n_obs
                    
                blaschke = blaschke * mobius(a_parameters[k], time_points)
                
    AFD2FMM_matrix = transition_matrix(a_parameters)
    for ch_i in range(n_ch):    
        phis[ch_i, 0:k+1] = np.dot(AFD2FMM_matrix, coefs[ch_i, :].T).T
        
    prediction, coefs = predict2(a_parameters, analytic_data_matrix, time_points)
    
    return a_parameters, coefs, phis, prediction
    


def opt_mobius_fun(arg, data_matrix, time_points, weights):
    ts = 2*np.arctan(arg[1]*np.tan((time_points[0] - arg[0])/2)) 
    DM = np.column_stack((np.ones(data_matrix.shape[1]), np.cos(ts), np.sin(ts)))
    linears = np.linalg.inv(DM.T @ DM) @ DM.T @ data_matrix.T
    #Weighted RSS
    return(np.sum([weights[ch]*np.sum((data_matrix[ch] - linears[0,ch] - linears[1,ch]*np.cos(ts) - linears[2,ch]*np.sin(ts))**2) for ch in range(data_matrix.shape[0])]))


def fit_fmm_k_mob(data_matrix, time_points=None, n_back=None, max_iter=1,
                  alpha_grid=None, omega_grid=None, 
                  weights=None, post_optimize=True, 
                  omega_min = 0.001, omega_max=1):
    
    n_ch, n_obs = data_matrix.shape
    
    # Grid definition.
    X, Y = np.meshgrid(alpha_grid, omega_grid.real)
    fmm_grid = np.column_stack((X.ravel(), Y.ravel()))
    RSS = np.zeros(fmm_grid.shape[0])
    # Parameters
    best_pars = [None] * n_back
    best_pars_linear = [None] * n_back
    components = [None] * n_back
    # Remainder
    remainder = np.copy(data_matrix)
    
    # Precalculations for each grid node:
    # t_star = 2*tan(omega(tan((t-alpha)/2)))
    # Dm = [1 cos(t_star) sin(t_star)]
    # OLS = inv(DM^T * DM) * DM^T * Y,  we precalculate: inv(DM^T * DM) * DM^T
    weights = 1/np.var(data_matrix, axis = 1)
     
    TS = [2*np.arctan(node[1]*np.tan((time_points[0] - node[0])/2)) for node in fmm_grid]
    cosTF = [np.cos(ts) for ts in TS]
    sinTF = [np.sin(ts) for ts in TS]
    DMs = [np.column_stack((np.ones(n_obs), cosTF[j], sinTF[j])) for j in range(len(TS))]
    precalculations = [np.linalg.inv(DM.T @ DM) @ DM.T for DM in DMs] # inv(X'X) X' 
    
    ## 1 Iteration of the backfitting algorithm: fit k waves
    for k in range(n_back):
        
        # GRID STEP
        estimates = [prec @ remainder.T for prec in precalculations]
        RSS = [sum([weights[ch]*np.sum((remainder[ch] - est[0,ch] - est[1,ch]*cosTF[j] - est[2,ch]*sinTF[j])**2) 
                    for ch in range(n_ch)]) 
               for j, est in enumerate(estimates)]
        min_index = np.argmin(RSS) 
        
        # OPTIMIZATION STEP
        if(post_optimize):
            res = minimize(opt_mobius_fun, x0=(fmm_grid[min_index]), 
                           args=(remainder, time_points, weights), 
                           method='L-BFGS-B', bounds=[(-2*np.pi, 4*np.pi),
                                                      (omega_min, omega_max)], 
                           tol=1e-4, options={'disp': False})
            best_pars[k] = res.x
        else:
            best_pars[k] = fmm_grid[min_index]
        
        # PREDICTION AND REMAINDER CALCULATIONS
        ts = 2*np.arctan(best_pars[k][1]*np.tan((time_points[0] - best_pars[k][0])/2)) 
        DM = np.column_stack((np.ones(ts.shape[0]), np.cos(ts), np.sin(ts)))
        linears = np.linalg.inv(DM.T @ DM) @ DM.T @ remainder.T
        best_pars_linear[k] = linears
        components[k] = np.column_stack([linears[0,ch] + linears[1,ch]*np.cos(ts) + linears[2,ch]*np.sin(ts) for ch in range(n_ch)]).T
        remainder = remainder - components[k]
        weights = 1/np.var(remainder, axis = 1)
        
    if max_iter > 1:
        for iter_j in range(1,max_iter):
            for k in range(n_back):
                
                # Repeat estimation for component k
                remainder = remainder + components[k]
                weights = 1/np.var(remainder, axis = 1)    
                
                # GRID STEP (Precalculated matrix*residuals)
                estimates = [prec @ remainder.T for prec in precalculations]
                RSS = [sum([weights[ch]*np.sum((remainder[ch] - est[0,ch] - est[1,ch]*np.cos(TS[j]) - est[2,ch]*np.sin(TS[j]))**2) 
                            for ch in range(n_ch)]) 
                       for j, est in enumerate(estimates)]
                min_index = np.argmin(RSS) 
                
                # OPTIMIZATION STEP
                if(post_optimize):
                    res = minimize(opt_mobius_fun, x0=(fmm_grid[min_index]), 
                                   args=(remainder, time_points, weights), 
                                   method='L-BFGS-B', bounds=[(-2*np.pi, 4*np.pi),
                                                              (omega_min, omega_max)], 
                                   tol=1e-4, options={'disp': False})
                    best_pars[k] = res.x
                else:
                    best_pars[k] = fmm_grid[min_index]
                
                # PREDICTION AND REMAINDER CALCULATIONS
                ts = 2*np.arctan(best_pars[k][1]*np.tan((time_points[0] - best_pars[k][0])/2)) 
                DM = np.column_stack((np.ones(ts.shape[0]), np.cos(ts), np.sin(ts)))
                linears = np.linalg.inv(DM.T @ DM) @ DM.T @ remainder.T
                best_pars_linear[k] = linears
                predictions = np.column_stack([linears[0,ch] + linears[1,ch]*np.cos(ts) + linears[2,ch]*np.sin(ts) for ch in range(n_ch)]).T
                remainder = remainder - predictions
                weights = 1/np.var(remainder, axis = 1)    
            
    return best_pars, best_pars_linear, remainder
    

def RSS_grid(data, est, cosTF, sinTF, weights):
    n_ch = est.shape[1]
    return sum([weights[ch]*np.sum((data[ch] - est[0,ch] - est[1,ch]*cosTF - est[2,ch]*sinTF)**2)  for ch in range(n_ch)])


