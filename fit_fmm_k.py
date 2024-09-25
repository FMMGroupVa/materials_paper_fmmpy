# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 09:46:50 2024

@author: Christian
"""

import numpy as np
from numpy.fft import fft, ifft
from scipy.optimize import minimize
from numba import jit
from auxiliar_functions import seq_times, szego, mobius, predict

@jit
def szego(a, t): 
    return ((1 - np.abs(a)**2) ** 0.5) / (1 - np.conj(a)*np.exp(1j*t))

@jit
def mobius(a, t): 
    return ((np.exp(1j*t) - a)) / (1 - np.conj(a)*np.exp(1j*t))

# No deja hacer jit, el problema es "float('inf')", quizás podríamos evitar 
# hacerlo de esta manera...
def inner_products_sum(splitted_a, analytic_data_matrix, t, weights):
    a = splitted_a[0]+1j*splitted_a[1]
    if abs(a)>=1:
        return float('inf')
    
    sum_abs = 0
    for ch_i in range(analytic_data_matrix.shape[0]):
        
        sum_abs = sum_abs + weights[ch_i]*(
            abs(np.conj(szego(a, t).dot(
                analytic_data_matrix[ch_i,:].conj().T)
                )) ** 2
            )
    return -sum_abs

# Pensaba que esta version podría ser un poco más rápida que la anterior,
# no hay diferencias
def inner_products_sum_2(splitted_a, analytic_data_matrix, t, weights):
    a = splitted_a[0]+1j*splitted_a[1]
    if abs(a)>=1:
        return float('inf')
    
    return -sum([weights[ch_i]*(abs(
        np.conj(szego(a, t).dot(
            analytic_data_matrix[ch_i,:].conj().T))) ** 2) 
        for ch_i in range(analytic_data_matrix.shape[0])])

@jit
def split_complex(z): 
    return ((z.real, z.imag))

def fit_fmm_k(analytic_data_matrix, time_points=None, n_back=None, 
              omega_grid=None, weights=None, post_optimize=True):
    
    n_ch, n_obs = analytic_data_matrix.shape
    
    # Grid definition.
    fmm_grid = np.meshgrid(omega_grid, time_points)
    afd_grid = (1-fmm_grid[0])/(1+fmm_grid[0])*np.exp(1j*(fmm_grid[1]))

    modules_grid = (1-omega_grid)/(1+omega_grid)*np.exp(1j*0)
    an_search_len = modules_grid.shape[0]
    
    # base: DFT coefficients of szego kernels with different a's (different 
    # modules)
    base = np.zeros((modules_grid.shape[0], n_obs), dtype=complex)
    for i in range(an_search_len):
        base[i,:] = fft(szego(modules_grid[i], time_points), n_obs)
    
    # Parameters (AFD)
    coefs = np.zeros((n_ch, n_back+1), dtype=complex)
    a_parameters = np.zeros(n_back+1, dtype=complex)

    # Start decomposing: c0 (mean)
    # Remainder R_(k+1) = ( R_k - c_k*e_ak(t) ) / m_ak(t)
    # e_ak := szego(ak, t)
    # m_ak := mobius(ak, t)
    
    remainder = np.copy(analytic_data_matrix)
    for ch_i in range(n_ch):
        coefs[ch_i,0] = np.mean(analytic_data_matrix[ch_i,:])
        remainder[ch_i,:] = ((analytic_data_matrix[ch_i,:] - coefs[ch_i,0])
                          /np.exp(1j*time_points))
    
    for k in range(n_back):
        abs_coefs = 0
        # Grid step
        for ch_i in range(n_ch):
            #abs_coefs += np.abs(ifft(pymat.repmat(fft(analytic_data_matrix[ch_i, :], n_obs), an_search_len, 1) * base, n_obs, 1))
            abs_coefs += np.abs(ifft(np.repeat(fft(
                remainder[ch_i, :], n_obs)[np.newaxis, :], 
                an_search_len, axis=0) * base, n_obs, 1))
        abs_coefs = abs_coefs.T
        
        # Best a
        max_loc_tmp = np.argwhere(abs_coefs == np.amax(abs_coefs))
        best_a = afd_grid[max_loc_tmp[0, 0], max_loc_tmp[0, 1]]

        # Nelder mead
        if(post_optimize):
            '''
            res = minimize(
                inner_products_sum_2, x0=split_complex(best_a), 
                args=(analytic_data_matrix, time_points, weights), 
                method='nelder-mead', options={'disp': False})
            
            # no parece que cambie el tiempo si cambio de método de optimizacion
            '''
            res = minimize(
                inner_products_sum_2, x0=split_complex(best_a), 
                args=(remainder, time_points, weights), 
                method='BFGS', options={'disp': False})
            
            opt_a = res.x[0] + 1j*res.x[1]
            a_parameters[k+1] = opt_a
        else:
            a_parameters[k+1] = best_a
        
        # Coefficient calculations 
        szego_a = szego(a_parameters[k+1], time_points)
        for ch_i in range(n_ch):    
            coefs[ch_i, k+1] = np.conj(szego_a.dot(remainder[ch_i,:].conj().T))/n_obs
            remainder[ch_i,:] = ((remainder[ch_i,:] - coefs[ch_i,k+1]*szego_a) 
                                 / mobius(a_parameters[k+1], time_points))

                
    
    return a_parameters, coefs, predict(a_parameters, coefs, time_points)
    


    
    

