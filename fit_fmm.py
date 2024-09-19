# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd 
import scipy.signal as sc
from auxiliar_functions import seq_times
import fit_fmm_unit

exc_data_1 = "'data_matrix' is not an instance of 'numpy.ndarray'"
exc_data_2 = "'data_matrix' must have 2 dimensions"
exc_omega = "Arguments error: Check that: 0 < omega_min < omega_max < 1"

def fit_fmm(data_matrix, time_points=None, n_back=1, max_iter=1, 
            omega_min=0.001, omega_max = 0.99, length_omega_grid=24, 
            omega_grid=None):
    
    if isinstance(data_matrix, pd.DataFrame): # From DataFrame to ndarray
        data_matrix = data_matrix.values
        
    if not isinstance(data_matrix, np.ndarray): # not an ndarray
        raise Exception(exc_data_1)
    else: 
        if not data_matrix.ndim == 2: # ndarray but incorrect dimensions
            raise Exception(exc_data_2)
        
    if omega_min<=0 or omega_max>=1: # omega in (0,1)
        raise Exception(exc_omega)
        
        
    
    nCh, n_obs = data_matrix.shape
    
    omega_grid = np.exp(np.linspace(np.log(omega_min), np.log(omega_max), 
                                   num=length_omega_grid))
    
    
    if time_points is None:
        time_points = seq_times(data_matrix.shape[1])
    analytic_data_matrix = sc.hilbert(data_matrix, axis = 1)
    

    fmm_grid = np.meshgrid(omega_grid, time_points)
    afd_grid = (1-fmm_grid[0])/(1+fmm_grid[0])*np.exp(1j*(fmm_grid[1]+np.pi))
    
    
    
    print('Todo bien')















