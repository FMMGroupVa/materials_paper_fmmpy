# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import scipy.signal as sc
from auxiliar_functions import seq_times
# from numpy.fft import fft

from fit_fmm_k import fit_fmm_k
from fit_fmm_k_restr import fit_fmm_k_restr_alpha_omega, fit_fmm_k_restr_betas, fit_fmm_k_restr_all_params

from auxiliar_functions import seq_times, szego, mobius, predict, predict2, transition_matrix, split_complex, inner_products_sum_2

from FMMModel import FMMModel


exc_data_1 = "'data_matrix' is not an instance of 'numpy.ndarray'."
exc_data_2 = "'data_matrix' must have 2 dimensions."
exc_omega_restr_1 = "Arguments error: Check that: 0 < 'omega_min' < 'omega_max' < 1."
exc_omega_restr_2 = "An arc for an individual omega is out of ['omega_min', 'omega_max']."
exc_beta_restr_1 = "Invalid beta range: both 'beta_min' and 'beta_max' must be either None or numeric values. Mixed types are not allowed."
exc_beta_restr_2 = "Invalid beta range: 'beta_min' and 'beta_max' may be between 0 and 2pi."
exc_beta_restr_3 = "Invalid beta range: arc('beta_min', 'beta_max')<pi."
exc_alpha_omega_restr_1 = "Arrays containing restrictions have different lengths."
algorithm_arguments = "'n_back' and 'max_iter' must be >1."
grid_arguments = "'omega_grid' may have positive length."
post_optimize_arguments = "'post_optimize' must be a logical value."

def fit_fmm(data_matrix, time_points=None, n_back=1, max_iter=1, post_optimize=True,
            omega_min=0.001, omega_max = 0.99, length_omega_grid=24, omega_grid=None,
            alpha_restrictions=None, omega_restrictions=None, group_restrictions=None, 
            beta_min=None, beta_max=None):
    """
    Fits a Frequency Modulated Möbius (FMM) model to a multivariate signal.
    
    This function performs FMM-based decomposition of analytic signals, optionally 
    incorporating restrictions on frequencies (omega), phases (alpha), and shape parameters (beta).
    
    Parameters
    ----------
    data_matrix : numpy.ndarray or pandas.DataFrame
        Input data matrix of shape (n_channels, n_timepoints). If a DataFrame is provided, it will be converted.
    
    time_points : array-like or None, optional
        Vector of time points corresponding to columns of `data_matrix`. If None, a default sequence is generated.
    
    n_back : int, optional
        Number of AFD terms to include in the decomposition (default is 1). Must be >=1.
    
    max_iter : int, optional
        Maximum number of iterations for the fitting algorithm (default is 1). Must be >=1.
    
    post_optimize : bool, optional
        If True, post-optimization of coefficients is performed (default is True).
    
    omega_min : float, optional
        Minimum value for the frequency search grid, must be in (0,1). Default is 0.001.
    
    omega_max : float, optional
        Maximum value for the frequency search grid, must be in (0,1). Default is 0.99.
    
    length_omega_grid : int, optional
        Number of points in the log-scale omega grid (if `omega_grid` is None). Must be > 0. Default is 24.
    
    omega_grid : array-like or None, optional
        Custom grid of omega values. If None, a default exponential grid is generated.
    
    alpha_restrictions : list of arrays or None, optional
        List of angle intervals (in radians) for each AFD term. Each element should be a tuple (alpha_min, alpha_max).
    
    omega_restrictions : list of arrays or None, optional
        List of frequency intervals for each AFD term. Each element should be a tuple (omega_min, omega_max).
    
    group_restrictions : list of int or None, optional
        Grouping indices for combining alpha and omega restrictions. Must match the length of restrictions if provided.
    
    beta_min : float or None, optional
        Minimum value for beta phase restriction (in radians). Must be in [0, 2π] if used.
    
    beta_max : float or None, optional
        Maximum value for beta phase restriction (in radians). Must be in [0, 2π] if used.
    
    Returns
    -------
    FMMModel
        An object containing the fitted model, including the estimated parameters and prediction.
    
    Raises
    ------
    Exception
        If inputs are not valid, including incompatible dimensions, invalid ranges, or inconsistent restrictions.
    
    Examples
    --------
    print("Por hacer")
    """
    if isinstance(data_matrix, pd.DataFrame): # From DataFrame to ndarray
        data_matrix = data_matrix.values
        
    if not isinstance(data_matrix, np.ndarray): # not an ndarray
        raise Exception(exc_data_1)
    else: 
        if not data_matrix.ndim == 2: # ndarray but incorrect dimensions
            raise Exception(exc_data_2)
        
    if (beta_min is None) ^ (beta_max is None):
        raise Exception(exc_beta_restr_1)
    
    if not beta_min is None:
        if beta_min < 0 or beta_min > 2*np.pi or beta_max < 0 or beta_max > 2*np.pi:
            raise Exception(exc_beta_restr_2)
            
        if (beta_max-beta_min)%(2*np.pi) > np.pi:
            raise Exception(exc_beta_restr_3)
    
    if (not alpha_restrictions is None) and (not omega_restrictions is None): 
        if group_restrictions is None:
            if not len(alpha_restrictions) == len(omega_restrictions):
                raise Exception(exc_alpha_omega_restr_1)
        else:
            if not len(group_restrictions) == len(alpha_restrictions) == len(omega_restrictions):
                raise Exception(exc_alpha_omega_restr_1)
    
    
    if omega_min<=0 or omega_max>=1: # omega in (0,1)
        raise Exception(exc_omega_restr_1)
        
    if not omega_restrictions is None:
        if sum([ome[0]>omega_max or ome[1]<omega_min for ome in omega_restrictions])>0:
            raise Exception(exc_omega_restr_2)
    
    if n_back < 1 or max_iter < 1:
        raise Exception(algorithm_arguments)
    

        
    if not isinstance(post_optimize, bool):
        raise Exception(post_optimize_arguments)
    n_ch, n_obs = data_matrix.shape
    
    if omega_grid is None:
        if length_omega_grid < 1:
            raise Exception(grid_arguments)
        omega_grid = np.exp(np.linspace(np.log(omega_min), np.log(omega_max), 
                                       num=length_omega_grid+2))[1:-2]
    
    if time_points is None:
        time_points = seq_times(data_matrix.shape[1])
    analytic_data_matrix = sc.hilbert(data_matrix, axis = 1)
    
    ###########################################################################
    
    # Parameters (AFD)
    if alpha_restrictions is None and omega_restrictions is None and beta_min is None and beta_max is None:
        restricted_flag = False
        a, coefs, phis, prediction = fit_fmm_k(analytic_data_matrix=analytic_data_matrix, 
                                               time_points=time_points, 
                                               n_back=n_back, max_iter=max_iter, 
                                               omega_grid=omega_grid, weights=np.ones(n_ch), post_optimize=True, 
                                               omega_min=omega_min, omega_max=omega_max)
    
    elif beta_min is None and beta_max is None:
        print("Restricted alphas-omegas")
        restricted_flag = True
        if group_restrictions is None:
            group_restrictions = [i for i in range(n_back)]
        a, coefs, phis, prediction = fit_fmm_k_restr_alpha_omega(analytic_data_matrix, time_points=time_points, n_back=n_back, max_iter=max_iter,
                                                     omega_grid=omega_grid, weights=np.ones(n_ch), post_optimize=post_optimize, 
                                                     omega_min=omega_min, omega_max=omega_max, 
                                                     alpha_restrictions=alpha_restrictions, omega_restrictions=omega_restrictions,
                                                     group_restrictions=group_restrictions)
        
    elif alpha_restrictions is None and omega_restrictions is None:
        print("Restricted betas")
        restricted_flag = True
        a, coefs, phis, prediction = fit_fmm_k_restr_betas(analytic_data_matrix, time_points=time_points, n_back=n_back, max_iter=max_iter, 
                                                           omega_grid=omega_grid, weights=np.ones(n_ch), post_optimize=post_optimize, 
                                                           omega_min=omega_min, omega_max=omega_max, 
                                                           beta_min=beta_min, beta_max=beta_max)
    
    else:
        print("All restricted")
        restricted_flag = True
        if group_restrictions is None:
            group_restrictions = [i for i in range(n_back)]
        a, coefs, phis, prediction = fit_fmm_k_restr_all_params(analytic_data_matrix, time_points=time_points, n_back=n_back, max_iter=max_iter, 
                                                           omega_grid=omega_grid, weights=np.ones(n_ch), post_optimize=post_optimize, 
                                                           omega_min=omega_min, omega_max=omega_max, 
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
    
    result = FMMModel(data=data_matrix, time_points=time_points, prediction=prediction, 
                      params=params, restricted=restricted_flag, max_iter=max_iter)
    
    return result


























