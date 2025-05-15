# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import scipy.signal as sc
from auxiliar_functions import seq_times
from numpy.fft import fft

from fit_fmm_k import fit_fmm_k
from fit_fmm_k_restr import fit_fmm_k_restr_alpha_omega, fit_fmm_k_restr_betas, fit_fmm_k_restr_all_params

from auxiliar_functions import seq_times, szego, mobius, predict, predict2, transition_matrix, split_complex, inner_products_sum_2

from FMMModel import FMMModel


exc_data_1 = "'data_matrix' is not an instance of 'numpy.ndarray'"
exc_data_2 = "'data_matrix' must have 2 dimensions"
exc_omega = "Arguments error: Check that: 0 < omega_min < omega_max < 1"

def fit_fmm(data_matrix, time_points=None, n_back=1, max_iter=1, post_optimize=True,
            omega_min=0.001, omega_max = 0.99, length_omega_grid=24, omega_grid=None,
            alpha_restrictions=None, omega_restrictions=None, group_restrictions=None, 
            beta_min=None, beta_max=None):
    
    if isinstance(data_matrix, pd.DataFrame): # From DataFrame to ndarray
        data_matrix = data_matrix.values
        
    if not isinstance(data_matrix, np.ndarray): # not an ndarray
        raise Exception(exc_data_1)
    else: 
        if not data_matrix.ndim == 2: # ndarray but incorrect dimensions
            raise Exception(exc_data_2)
        
    if omega_min<=0 or omega_max>=1: # omega in (0,1)
        raise Exception(exc_omega)
    
    n_ch, n_obs = data_matrix.shape
    
    omega_grid = np.exp(np.linspace(np.log(omega_min), np.log(omega_max), 
                                   num=length_omega_grid+2))[1:-2]
    
    if time_points is None:
        time_points = seq_times(data_matrix.shape[1])
    analytic_data_matrix = sc.hilbert(data_matrix, axis = 1)
    
    ###########################################################################
    
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
    if alpha_restrictions is None and omega_restrictions is None and beta_min is None and beta_max is None:
    
        a, coefs, phis, prediction = fit_fmm_k(analytic_data_matrix=analytic_data_matrix, 
                                               time_points=time_points, 
                                               n_back=n_back, max_iter=max_iter, 
                                               omega_grid=omega_grid, weights=np.ones(n_ch), post_optimize=True, 
                                               omega_min=omega_min, omega_max=omega_max)
    
    elif beta_min is None and beta_max is None:
        print("Restricted alphas-omegas")
        if group_restrictions is None:
            group_restrictions = [i for i in range(n_back)]
        a, coefs, phis, prediction = fit_fmm_k_restr_alpha_omega(analytic_data_matrix, time_points=time_points, n_back=n_back, max_iter=max_iter,
                                                     omega_grid=omega_grid, weights=np.ones(n_ch), post_optimize=post_optimize, 
                                                     omega_min=omega_min, omega_max=omega_max, 
                                                     alpha_restrictions=alpha_restrictions, omega_restrictions=omega_restrictions,
                                                     group_restrictions=group_restrictions)
        
    elif alpha_restrictions is None and omega_restrictions is None:
        print("Restricted betas")
        a, coefs, phis, prediction = fit_fmm_k_restr_betas(analytic_data_matrix, time_points=time_points, n_back=n_back, max_iter=max_iter, 
                                                           omega_grid=omega_grid, weights=np.ones(n_ch), post_optimize=post_optimize, 
                                                           omega_min = omega_min, omega_max=omega_max, 
                                                           beta_min=beta_min, beta_max=beta_max)
    
    else:
        print("All restricted")
        if group_restrictions is None:
            group_restrictions = [i for i in range(n_back)]
        a, coefs, phis, prediction = fit_fmm_k_restr_all_params(analytic_data_matrix, time_points=time_points, n_back=n_back, max_iter=max_iter, 
                                                           omega_grid=omega_grid, weights=np.ones(n_ch), post_optimize=post_optimize, 
                                                           omega_min = omega_min, omega_max=omega_max, 
                                                           alpha_restrictions=alpha_restrictions, omega_restrictions=omega_restrictions,
                                                           group_restrictions=group_restrictions,
                                                           beta_min=beta_min, beta_max=beta_max)
    alphas = (np.angle(a[1:]) + np.pi) % (2*np.pi)
    As = np.abs(phis[:,1:])
    betas = (np.angle(phis[:,1:]) - alphas + np.pi) % (2*np.pi)
    params = {'alpha': alphas,
              'omega': (1-np.abs(a[1:]))/(1+np.abs(a[1:])),
              'a': a,
              'M': np.abs(phis[:,0]),
              'A': As, 
              'beta': betas,
              'delta': As*np.cos(betas),
              'gamma': As*np.sin(betas),
              'coef': coefs,
              'phi': phis}
        
    result = FMMModel(data=data_matrix, time_points=time_points, prediction=prediction, params=params)
    
    return result


























