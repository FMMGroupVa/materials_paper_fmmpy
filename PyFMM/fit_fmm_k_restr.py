# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 10:07:03 2025

@author: Christian
"""

import numpy as np
from numpy.fft import fft, ifft
from scipy.optimize import minimize
# from numba import jit
from .auxiliar_functions import szego, mobius, predict, predict2, transition_matrix, inner_products_sum_2, split_complex #, seq_times
from qpsolvers import solve_ls #, solve_qp

def RSS_restr_betas(splitted_a, data_matrix, time_points, k, a_parameters, weights, beta_restrictions):
    a_parameters[k] = splitted_a[1]*np.exp(1j*splitted_a[0])
    return project_betas_2(data_matrix, weights, time_points, a_parameters, beta_restrictions)

# Generates a matrix with p restrictions (between a and b)
def generate_G(p, a, b):
    G = np.zeros((2*p, 2*p+1))
    for var in range(p):
        G[2*var, 2*var+1] = np.sin(a)
        G[2*var, 2*var+2] = np.cos(a)
        G[2*var+1, 2*var+1] = -np.sin(b)
        G[2*var+1, 2*var+2] = -np.cos(b)
    return G

# Generates a matrix with p restrictions (between ak and bk)
def generate_G_ch(p, beta_restrictions_ch):
    
    n_restr = sum(1 for bmin, _ in beta_restrictions_ch if bmin is not None)
    G = np.zeros((2*n_restr, 2*p+1))
    row_i = 0
    for var in range(p):
        bmin, bmax = beta_restrictions_ch[var]
        if bmin is None or bmax is None:
            continue  
        G[2 * row_i, 2 * var + 1] =  np.sin(bmin)
        G[2 * row_i, 2 * var + 2] =  np.cos(bmin)
        G[2 * row_i + 1, 2 * var + 1] = -np.sin(bmax)
        G[2 * row_i + 1, 2 * var + 2] = -np.cos(bmax)
        row_i += 1
    return G

def project_betas(data_matrix, time_points, a, beta_restrictions):
    
    n_back = len(a) - 1
    n_ch, n_obs = data_matrix.shape
    
    h = np.zeros(2 * n_back)
    
    # 2. Design matrix 
    alphas = np.angle(a[1:]) + np.pi
    omegas = (1 - np.abs(a[1:])) / (1 + np.abs(a[1:]))
    
    ts = [2*np.arctan(omegas[i] * np.tan((time_points[0] - alphas[i])/2)) for i in range(n_back)]
    DM = np.column_stack([np.ones(n_obs)] + [np.column_stack([np.cos(ts[i]), np.sin(ts[i])]) for i in range(n_back)])
    
    # 4. Allocate storage
    RLS = np.zeros((n_ch, 2 * n_back + 1))
    phis = np.zeros((n_ch, n_back + 1), dtype=np.complex128)
    
    # 5. Solve LSQ problem for all channels
    for ch_i in range(n_ch):
        G = generate_G_ch(n_back, beta_restrictions[ch_i])
        RLS[ch_i] = solve_ls(DM, data_matrix[ch_i], G=G, h=h, solver='quadprog')
        # RLS[ch_i] = solve_qp(DM.T@DM, -DM.T@data_matrix[ch_i], G=G, h=h, solver='quadprog')
    
    # 6. Compute betas, amplitudes, and phis using vectorized operations
    betas = np.arctan2(-RLS[:, 2::2], RLS[:, 1::2]) % (2*np.pi)
    amplitudes = np.sqrt(RLS[:, 1::2] ** 2 + RLS[:, 2::2] ** 2)
    phis[:, 0] = RLS[:, 0]
    phis[:, 1:] = amplitudes * np.exp(1j * (betas - alphas))

    # Return AFD coefs
    return np.dot(np.linalg.inv(transition_matrix(a)), phis.T).T

def project_betas_2(data_matrix, weights, time_points, a, beta_restrictions):
    
    n_back = len(a) - 1
    if len(beta_restrictions[0])!=n_back:
        print("Dimension error.")
        
    n_ch, n_obs = data_matrix.shape

    h = np.zeros(2 * n_back)
    
    # 2. Design matrix 
    alphas = np.angle(a[1:]) + np.pi
    omegas = (1 - np.abs(a[1:])) / (1 + np.abs(a[1:]))
    
    ts = [2*np.arctan(omegas[i] * np.tan((time_points[0] - alphas[i])/2)) for i in range(n_back)]
    DM = np.column_stack([np.ones(n_obs)] + [np.column_stack([np.cos(ts[i]), np.sin(ts[i])]) for i in range(n_back)])
    
    # 4. Allocate storage
    RLS = np.zeros((n_ch, 2 * n_back + 1))
    
    # 5. Solve LSQ problem for all channels
    for ch_i in range(n_ch):
        G = generate_G_ch(n_back, beta_restrictions[ch_i])
        RLS[ch_i] = solve_ls(DM, data_matrix[ch_i], G=G, h=h, solver='quadprog')
        
        # RLS[ch_i] = solve_qp(DM.T @ DM, -DM.T@data_matrix[ch_i], G=G, h=h, solver='quadprog')
    
    prediction = np.dot(DM, RLS.T)
    res_sq = (data_matrix - prediction.T)**2
    rss_ch = np.sum(res_sq, axis=1)  # shape (4,)
    weighted_rss = np.sum(weights * rss_ch)
    
    # Return AFD coefs
    return weighted_rss

def fit_fmm_k_restr_alpha_omega(analytic_data_matrix, time_points=None, n_back=None, max_iter=None,
                    omega_grid=None, weights=None, post_optimize=True, 
                    omega_min = 0.001, omega_max=0.999, 
                    alpha_restrictions=None, omega_restrictions=None, 
                    group_restrictions=None):
    
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
    alpha_restrictions_2 = [((alpha[0] + np.pi) % (2*np.pi), (alpha[1] + np.pi) % (2*np.pi)) for alpha in alpha_restrictions]
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
    data_norm = np.zeros(n_ch)
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
        data_norm[ch_i] = np.var((analytic_data_matrix[ch_i,:] - coefs[ch_i,0]).real)
        weights[ch_i] = 1/data_norm[ch_i] # sigmas^2
    
    ## 1 Iteration of the backfitting algorithm: fit k waves
    # for k in range(1, n_back+1):
        
    unique_groups = sorted(set(group_restrictions))
    for k_idx, k_val in enumerate(unique_groups):
        indices_k = [i for i, g in enumerate(group_restrictions) if g == k_val]
        k = k_idx+1
        
        best_sum_abs_coef = -np.inf
        best_a = None
            
        for i in indices_k:
            # 1. Aplicar restricción omega: restringir basek, afd_grid2 y abs_coefs
            if omega_restrictions is not None:
                basek = base[(omega_grid > omega_restrictions[i][0]) & (omega_grid < omega_restrictions[i][1])]
                afd_grid2 = afd_grid[:, (omega_grid > omega_restrictions[i][0]) & (omega_grid < omega_restrictions[i][1])]
            else:
                basek = base
                afd_grid2 = afd_grid
                
            abs_coefs = 0
            for ch_i in range(n_ch):
                abs_coefs += weights[ch_i] * np.abs(ifft(
                    np.repeat(fft(remainder[ch_i, :], n_obs)[np.newaxis, :], basek.shape[0], axis=0) * basek, n_obs, 1))
    
            abs_coefs = abs_coefs.T  # n_obs x n_omegas

            # Best a: we only select alphas in the restricted arc
            if omega_restrictions is not None:
                afd_grid2 = afd_grid[:,(omega_grid>omega_restrictions[i][0]) & (omega_grid<omega_restrictions[i][1])]
                
            if(alpha_restrictions_2[i][0] > alpha_restrictions_2[i][1]):
                # +++]--------[+++
                abs_coefs = abs_coefs[(time_points[0]>=alpha_restrictions_2[i][0]) | (time_points[0]<=alpha_restrictions_2[i][1]) ]
                afd_grid2 = afd_grid2[(time_points[0]>=alpha_restrictions_2[i][0]) | (time_points[0]<=alpha_restrictions_2[i][1]) ]
            else:
                # ------[++++]----
                abs_coefs = abs_coefs[(time_points[0]>=alpha_restrictions_2[i][0]) & (time_points[0]<=alpha_restrictions_2[i][1]) ]
                afd_grid2 = afd_grid2[(time_points[0]>=alpha_restrictions_2[i][0]) & (time_points[0]<=alpha_restrictions_2[i][1]) ]
            
            max_loc_tmp = np.argwhere(abs_coefs == np.amax(abs_coefs))
            best_a_tmp = afd_grid2[max_loc_tmp[0, 0], max_loc_tmp[0, 1]]
    
            ## STEP 2: Postoptimization - Profile log-likelihood.
            if(post_optimize):
                # We transform time points as: ---[+++]-----  ->  [+++]--------
                # (Easier way to impose arc restrictions in circular parameters)
                time_points_transformed = time_points - alpha_restrictions_2[i][0]
                # Lower values than the general omega_min are not allowed
                if omega_restrictions is not None:
                    omega_min_opt = max(omega_min, omega_restrictions[i][0])
                    omega_max_opt = min(omega_max, omega_restrictions[i][1])
                else:
                    omega_min_opt = omega_min
                    omega_max_opt = omega_max
                best_a_tmp = best_a_tmp*np.exp(-1j*alpha_restrictions_2[i][0])
                # Optimization routine
                res = minimize(
                    inner_products_sum_2, x0=split_complex(best_a_tmp), 
                    args=(remainder, time_points_transformed, weights), 
                    method='L-BFGS-B', 
                    bounds=[(0, # alphamin - alphamin
                            (alpha_restrictions_2[i][1] - alpha_restrictions_2[i][0]) % (2*np.pi)), # alphamax - alphamin
                            ((1-omega_max_opt)/(1+omega_max_opt), 
                             (1-omega_min_opt)/(1+omega_min_opt))],
                    tol=1e-4, options={'disp': False})
                opt_a = res.x[1]*np.exp(1j*res.x[0])
                best_a_tmp = opt_a * np.exp(1j*alpha_restrictions_2[i][0]) # alpha2 + alphamin
 
            # Coefficient calculations 
            szego_a = szego(best_a_tmp, time_points)
            sum_abs_coefs = 0
            for ch_i in range(n_ch):
                sum_abs_coefs =+ weights[ch_i]*np.abs(np.conj(szego_a.dot(remainder[ch_i,:].conj().T))/n_obs)[0]
            
            if sum_abs_coefs > best_sum_abs_coef:
                best_sum_abs_coef = sum_abs_coefs
                best_a = best_a_tmp
            
        a_parameters[k] = best_a
        
        szego_a = szego(a_parameters[k], time_points)
        for ch_i in range(n_ch):
            coefs[ch_i, k] = np.conj(szego_a.dot(remainder[ch_i,:].conj().T))/n_obs
            remainder[ch_i,:] = ((remainder[ch_i,:] - coefs[ch_i,k]*szego_a) 
                                 / mobius(a_parameters[k], time_points))
        
    if max_iter > 1:
        for iter_j in range(1, max_iter):
            # Auxiliar Blaschke product: z*Bl_{a_1,...,a_K} = z*m(a1,t)*...*m(aK,t)
            blaschke = z 
            for k in range(1, n_back+1):
                blaschke = blaschke*mobius(a_parameters[k], time_points)
        
            # for k in range(1, n_back+1):
            for k_idx, k_val in enumerate(unique_groups):
                indices_k = [i for i, g in enumerate(group_restrictions) if g == k_val]
                k = k_idx+1
                
                best_sum_abs_coef = -np.inf
                best_a = None
                
                # Calculate the standard reminder (data-prediction) without component k:  r = X - sum ci*Bi, i != k
                # std_remainder = analytic_data_matrix - predict(np.delete(a_parameters, k, axis=0), np.delete(coefs, k, axis=1), time_points)
                std_remainder = analytic_data_matrix - predict2(np.delete(a_parameters, k, axis=0), analytic_data_matrix, time_points)[0]
                
                # weights = data_norm - np.sum(np.abs(np.delete(coefs, [0, k], axis=1))**2, axis=1)/2
                weights = 1/np.var(std_remainder.real, axis=1)
                
                # Calculate the reduced reminder reminder/(z*mob1*...,mobK) (without k)
                blaschke = blaschke / mobius(a_parameters[k], time_points)
                # Reduced reminder (to calculate )
                remainder = std_remainder / blaschke
                
                
                for i in indices_k:
                    # Select basis components whose abs(a) / omegas are in the restricted range
                    
                    if omega_restrictions is not None:
                        basek = base[(omega_grid > omega_restrictions[i][0]) & (omega_grid < omega_restrictions[i][1])]
                        afd_grid2 = afd_grid[:, (omega_grid > omega_restrictions[i][0]) & (omega_grid < omega_restrictions[i][1])]
    
                    ## STEP 1: Grid search - AFD-FFT formulations
                    abs_coefs = 0
                    for ch_i in range(n_ch):
                        abs_coefs += weights[ch_i]*np.abs(ifft(
                            np.repeat(fft(remainder[ch_i, :], n_obs)[np.newaxis, :], basek.shape[0], axis=0) * basek, n_obs, 1))
                    abs_coefs = abs_coefs.T
                    
                    # Best a: we only select alphas in the restricted arc
                    afd_grid2 = afd_grid[:,(omega_grid>omega_restrictions[i][0]) & (omega_grid<omega_restrictions[i][1])]
                    if(alpha_restrictions_2[i][0] > alpha_restrictions_2[i][1]):
                        # +++]--------[+++  -> restricted alphas arc crossing 0 alphaMin > alphaMax
                        abs_coefs = abs_coefs[(time_points[0]>=alpha_restrictions_2[i][0]) | (time_points[0]<=alpha_restrictions_2[i][1])]
                        afd_grid2 = afd_grid2[(time_points[0]>=alpha_restrictions_2[i][0]) | (time_points[0]<=alpha_restrictions_2[i][1])]
                    else:
                        # ------[++++]----  -> restricted alphas in [alphaMin, alphaMax] alphaMin < alphaMax
                        abs_coefs = abs_coefs[(time_points[0]>=alpha_restrictions_2[i][0]) & (time_points[0]<=alpha_restrictions_2[i][1])]
                        afd_grid2 = afd_grid2[(time_points[0]>=alpha_restrictions_2[i][0]) & (time_points[0]<=alpha_restrictions_2[i][1])]
                        
                    max_loc_tmp = np.argwhere(abs_coefs == np.amax(abs_coefs))
                    best_a_tmp = afd_grid2[max_loc_tmp[0, 0], max_loc_tmp[0, 1]]
                    
                    ## STEP 2: Postoptimization - Profile log-likelihood.
                    if(post_optimize):
                        # We transform time points as: ---[+++]-----  ->  [+++]--------
                        time_points_transformed = time_points - alpha_restrictions_2[i][0] 
                        # Lower values than the general omega_min are not allowed
                        if omega_restrictions is not None:
                            omega_min_opt = max(omega_min, omega_restrictions[i][0])
                            omega_max_opt = min(omega_max, omega_restrictions[i][1])
                        else:
                            omega_min_opt = omega_min
                            omega_max_opt = omega_max
                        # Optimization routine
                        res = minimize(
                            inner_products_sum_2, x0=split_complex(best_a_tmp), 
                            args=(remainder, time_points_transformed, weights), 
                            method='L-BFGS-B',
                            bounds=[(0, # alphamin - alphamin
                                    (alpha_restrictions_2[i][1] - alpha_restrictions_2[i][0]) % (2*np.pi)), # alphamax - alphamin
                                    ((1-omega_max_opt)/(1+omega_max_opt), 
                                     (1-omega_min_opt)/(1+omega_min_opt))],
                            tol=1e-4, options={'disp': False})
                        opt_a = res.x[1]*np.exp(1j*res.x[0])
                        best_a_tmp = opt_a * np.exp(1j*alpha_restrictions_2[i][0]) # alpha + alphamin
                        
                    # Coefficient calculations 
                    szego_a = szego(best_a_tmp, time_points)
                    sum_abs_coefs = 0
                    for ch_i in range(n_ch):
                        sum_abs_coefs =+ weights[ch_i]*np.abs(np.conj(szego_a.dot(remainder[ch_i,:].conj().T))/n_obs)[0]
                    
                    if sum_abs_coefs > best_sum_abs_coef:
                        best_sum_abs_coef = sum_abs_coefs
                        best_a = best_a_tmp
                    
                a_parameters[k] = best_a
                
                # Coefficient calculations c_k = <Remainder, szego(a)>
                szego_a = szego(a_parameters[k], time_points)
                for ch_i in range(n_ch):
                    coefs[ch_i, k] = np.conj(szego_a.dot(remainder[ch_i,:].conj().T))/n_obs
                
                # Blaschke product update
                blaschke = blaschke * mobius(a_parameters[k], time_points)
                
    AFD2FMM_matrix = transition_matrix(a_parameters)
      
    prediction, coefs = predict2(a_parameters, analytic_data_matrix, time_points)
    phis = np.dot(AFD2FMM_matrix, coefs.T).T
    
    
    return a_parameters, coefs, phis, prediction


def fit_fmm_k_restr_betas(analytic_data_matrix, time_points=None, n_back=None, max_iter=None,
              alpha_grid=None, omega_grid=None, weights=None, post_optimize=True, omega_min=0.001, omega_max=0.99,
              beta_restrictions=None):
    
    if(analytic_data_matrix.ndim == 2):
        n_ch, _ = analytic_data_matrix.shape
    elif(analytic_data_matrix.ndim == 1):
        # n_obs = analytic_data_matrix.shape[0]
        analytic_data_matrix = analytic_data_matrix[np.newaxis, :]
    else:
        print("Bad data matrix dimensions.")
    
    if(max_iter==None):
        max_iter=1
    
    # Grid definition.
    fmm_grid = np.meshgrid(omega_grid, alpha_grid)
    afd_grid = (1-fmm_grid[0])/(1+fmm_grid[0])*np.exp(1j*(fmm_grid[1]))
    
    a_parameters = np.zeros(n_back+1, dtype=complex)
    weights = 1/np.var(analytic_data_matrix.real, axis=1, ddof=1)
    
    ## 1 Iteration of the backfitting algorithm: fit k waves
    for k in range(1, n_back+1):
        print((0,k))
        abs_coefs_2 = np.zeros((len(alpha_grid), len(omega_grid))) #alphas x omegas
        beta_restrictions_k = [row[:k] for row in beta_restrictions]
        
        for index, value in enumerate(afd_grid.ravel()):
            i, j = np.unravel_index(index, abs_coefs_2.shape)
            if not np.isin(value, a_parameters):
                a_parameters[k] = value
                abs_coefs_2[i,j] = project_betas_2(analytic_data_matrix.real, weights, time_points, a_parameters[:k+1], beta_restrictions_k)
            else:
                abs_coefs_2[i,j] = float('Inf')
        
        min_loc_tmp = np.argwhere(abs_coefs_2 == np.amin(abs_coefs_2))
        best_a = afd_grid[min_loc_tmp[0, 0], min_loc_tmp[0, 1]]
        
        a_parameters[k] = best_a
        print(a_parameters)
        if(post_optimize):
            res = minimize(
                RSS_restr_betas, x0=split_complex(best_a), 
                args=(analytic_data_matrix.real, time_points, k, a_parameters[:k+1], weights,  beta_restrictions_k), 
                # Bounds: (-2pi, 4pi) para explorar bien parametro circular
                method='L-BFGS-B', bounds=[(-2*np.pi, 4*np.pi), ((1-omega_max)/(1+omega_max), (1-omega_min)/(1+omega_min))],
                tol=1e-4, options={'disp': False})
            opt_a = res.x[1]*np.exp(1j*res.x[0])
            a_parameters[k] = opt_a
        
        coefs_proj = project_betas(analytic_data_matrix.real, time_points, a_parameters[:k+1],  beta_restrictions_k)
        std_remainder = analytic_data_matrix - predict(a_parameters[:k+1], coefs_proj, time_points)
        weights = 1/np.var(std_remainder, axis=1, ddof=1)
    
    if max_iter > 1:
        for iter_j in range(1,max_iter):
            for k in range(1, n_back+1):
                print((iter_j,k))
                abs_coefs_2 = np.full((len(alpha_grid), len(omega_grid)), np.inf)
                
                for index, value in enumerate(afd_grid.ravel()):
                    i, j = np.unravel_index(index, afd_grid.shape)
                    if not np.any(np.isclose(value, a_parameters, atol=1e-10, rtol=1e-5)):
                        a_parameters[k] = value
                        abs_coefs_2[i,j] = project_betas_2(analytic_data_matrix.real, weights, time_points, a_parameters, beta_restrictions)
                    else:
                        abs_coefs_2[i,j] = float('Inf')

                min_loc_tmp = np.argwhere(abs_coefs_2 == np.amin(abs_coefs_2))
                best_a = afd_grid[min_loc_tmp[0, 0], min_loc_tmp[0, 1]]
                
                a_parameters[k] = best_a

                # STEP 2: Postoptimization - Profile log-likelihood.
                if(post_optimize):
                    res = minimize(
                        RSS_restr_betas, x0=split_complex(best_a), 
                        args=(analytic_data_matrix, time_points, k, a_parameters, weights, beta_restrictions), 
                        # Bounds: (-2pi, 4pi) para explorar bien parametro circular
                        method='L-BFGS-B', bounds=[(-2*np.pi, 4*np.pi), ((1-omega_max)/(1+omega_max), (1-omega_min)/(1+omega_min))],
                        tol=1e-4, options={'disp': False})
                    opt_a = res.x[1]*np.exp(1j*res.x[0])
                    a_parameters[k] = opt_a
                
                coefs_proj = project_betas(analytic_data_matrix.real, time_points, a_parameters, beta_restrictions)
                std_remainder = analytic_data_matrix - predict(a_parameters, coefs_proj, time_points)
                weights = 1/np.var(std_remainder, axis=1, ddof=1)

    prediction = predict(a_parameters, coefs_proj, time_points)
    AFD2FMM_matrix = transition_matrix(a_parameters)
    phis = np.dot(AFD2FMM_matrix, coefs_proj.T).T
    
    return a_parameters, coefs_proj, phis, prediction



def fit_fmm_k_restr_all_params(analytic_data_matrix, time_points=None, n_back=None, max_iter=None,
              alpha_grid=None, omega_grid=None, weights=None, post_optimize=True, omega_min=0.001, omega_max=0.99, 
              alpha_restrictions=None, omega_restrictions=None, group_restrictions=None,
              beta_restrictions=None):
    
    if(analytic_data_matrix.ndim == 2):
        n_ch, n_obs = analytic_data_matrix.shape
    elif(analytic_data_matrix.ndim == 1):
        # n_obs = analytic_data_matrix.shape[0]
        analytic_data_matrix = analytic_data_matrix[np.newaxis, :]
    else:
        print("Bad data matrix dimensions.")
    
    if(max_iter==None):
        max_iter=1
        
    alpha_restrictions_2 = [((alpha[0] + np.pi) % (2*np.pi), (alpha[1] + np.pi) % (2*np.pi)) for alpha in alpha_restrictions]
    alpha_grid_2 = (alpha_grid + np.pi) % (2*np.pi)
    
    # Grid definition.
    fmm_grid = np.meshgrid(omega_grid, alpha_grid_2)
    afd_grid = (1-fmm_grid[0])/(1+fmm_grid[0])*np.exp(1j*(fmm_grid[1]))
    a_parameters = np.zeros(n_back+1, dtype=complex)

    weights = 1/np.var(analytic_data_matrix.real, axis=1, ddof=1)
    
    ## 1 Iteration of the backfitting algorithm: fit k waves
    unique_groups = sorted(set(group_restrictions))
    for k_idx, k_val in enumerate(unique_groups):
        indices_k = [i for i, g in enumerate(group_restrictions) if g == k_val]
        k = k_idx+1
        
        best_sum_RSS = np.inf
        best_a = None
        print((0,k))
        
        for i in indices_k:
            
            beta_restrictions_k = [row[:k] for row in beta_restrictions]
            # Best a: we only select alphas in the restricted arc
            if omega_restrictions is not None:
                afd_grid2 = afd_grid[:,(omega_grid>omega_restrictions[i][0]) & (omega_grid<omega_restrictions[i][1])]
            else:
                afd_grid2 = afd_grid

            if(alpha_restrictions_2[i][0] > alpha_restrictions_2[i][1]):
                # +++]--------[+++
                mask_alpha = (alpha_grid_2 >= alpha_restrictions_2[i][0]) | (alpha_grid_2 <= alpha_restrictions_2[i][1])
            else:
                # ------[++++]----
                mask_alpha = (alpha_grid_2 >= alpha_restrictions_2[i][0]) & (alpha_grid_2 <= alpha_restrictions_2[i][1])
            afd_grid2 = afd_grid2[mask_alpha, :]
            
            abs_coefs = np.zeros(afd_grid2.shape) #alphas x omegas
            
            for index, value in enumerate(afd_grid2.ravel()):
                i2, j2 = np.unravel_index(index, afd_grid2.shape)
                if not np.any(np.isclose(value, a_parameters, atol=1e-10, rtol=1e-5)):
                    a_parameters[k] = value
                    abs_coefs[i2,j2] = project_betas_2(analytic_data_matrix.real, weights, time_points, a_parameters[:k+1], beta_restrictions_k)
                else:
                    abs_coefs[i2,j2] = float('Inf')

            min_loc_tmp = np.argwhere(abs_coefs == np.amin(abs_coefs))
            best_a_tmp = afd_grid2[min_loc_tmp[0, 0], min_loc_tmp[0, 1]]
            ## STEP 2: Postoptimization - Profile log-likelihood.
            if(post_optimize):
                # We transform time points as: ---[+++]-----  ->  [+++]--------
                # (Easier way to impose arc restrictions in circular parameters)
                time_points_transformed = time_points - alpha_restrictions_2[i][0]
                # Lower values than the general omega_min are not allowed
                if omega_restrictions is not None:
                    omega_min_opt = max(omega_min, omega_restrictions[i][0])
                    omega_max_opt = min(omega_max, omega_restrictions[i][1])
                else:
                    omega_min_opt = omega_min
                    omega_max_opt = omega_max
                    
                best_a_tmp = best_a_tmp*np.exp(-1j*alpha_restrictions_2[i][0])
                # Optimization routine
                res = minimize(
                    RSS_restr_betas, x0=split_complex(best_a_tmp), 
                    args=(analytic_data_matrix.real, time_points_transformed, k, a_parameters[:k+1]*np.exp(-1j*alpha_restrictions_2[i][0]), weights, beta_restrictions_k),
                    method='L-BFGS-B', 
                    bounds=[(0, # alphamin - alphamin
                            (alpha_restrictions_2[i][1] - alpha_restrictions_2[i][0]) % (2*np.pi)), # alphamax - alphamin
                            ((1-omega_max_opt)/(1+omega_max_opt), 
                              (1-omega_min_opt)/(1+omega_min_opt))],
                    tol=1e-4, options={'disp': False})
                best_a_tmp = res.x[1]*np.exp(1j*res.x[0])*np.exp(1j*alpha_restrictions_2[i][0]) # alpha2 + alphamin
                
            a_parameters_tmp = a_parameters[:k+1]
            a_parameters_tmp[k] = best_a_tmp
            
            sum_RSS = project_betas_2(analytic_data_matrix.real, weights, time_points, a_parameters_tmp, beta_restrictions_k)
            
            if sum_RSS < best_sum_RSS:
                best_sum_RSS = sum_RSS
                best_a = best_a_tmp
        a_parameters[k] = best_a
            
    coefs_proj = project_betas(analytic_data_matrix.real, time_points, a_parameters, beta_restrictions)
    
    print(a_parameters)
    
    if max_iter > 1:
        for iter_j in range(1,max_iter):
            for k_idx, k_val in enumerate(unique_groups):
                print((iter_j,k))
                indices_k = [i for i, g in enumerate(group_restrictions) if g == k_val]
                k = k_idx+1
                
                best_sum_RSS = np.inf
                best_a = None
                
                for i in indices_k:
                    
                    # Best a: we only select alphas in the restricted arc
                    if omega_restrictions is not None:
                        afd_grid2 = afd_grid[:,(omega_grid>omega_restrictions[i][0]) & (omega_grid<omega_restrictions[i][1])]
                    else:
                        afd_grid2 = afd_grid
                        
                    if(alpha_restrictions_2[i][0] > alpha_restrictions_2[i][1]):
                        # +++]--------[+++
                        mask_alpha = (alpha_grid_2 >= alpha_restrictions_2[i][0]) | (alpha_grid_2 <= alpha_restrictions_2[i][1])
                    else:
                        # ------[++++]----
                        mask_alpha = (alpha_grid_2 >= alpha_restrictions_2[i][0]) & (alpha_grid_2 <= alpha_restrictions_2[i][1])
                    afd_grid2 = afd_grid2[mask_alpha, :]
                        
                    abs_coefs = np.zeros(afd_grid2.shape) #alphas x omegas

                    for index, value in enumerate(afd_grid2.ravel()):
                        i2, j2 = np.unravel_index(index, afd_grid2.shape)
                        if not np.any(np.isclose(value, a_parameters, atol=1e-10, rtol=1e-5)):
                            a_parameters[k] = value
                            abs_coefs[i2,j2] = project_betas_2(analytic_data_matrix.real, weights, time_points, a_parameters, beta_restrictions)
                        else:
                            abs_coefs[i2,j2] = float('Inf')
                    min_loc_tmp = np.argwhere(abs_coefs == np.amin(abs_coefs))
                    best_a_tmp = afd_grid2[min_loc_tmp[0, 0], min_loc_tmp[0, 1]]

                    ## STEP 2: Postoptimization - Profile log-likelihood.
                    if(post_optimize):
                        # We transform time points as: ---[+++]-----  ->  [+++]--------
                        # (Easier way to impose arc restrictions in circular parameters)
                        time_points_transformed = time_points - alpha_restrictions_2[i][0]
                        # Lower values than the general omega_min are not allowed
                        if omega_restrictions is not None:
                            omega_min_opt = max(omega_min, omega_restrictions[i][0])
                            omega_max_opt = min(omega_max, omega_restrictions[i][1])
                        else:
                            omega_min_opt = omega_min
                            omega_max_opt = omega_max
                            
                        best_a_tmp = best_a_tmp*np.exp(-1j*alpha_restrictions_2[i][0])
                        # Optimization routine
                        res = minimize(
                            RSS_restr_betas, x0=split_complex(best_a_tmp), 
                            args=(analytic_data_matrix.real, time_points_transformed, k, a_parameters*np.exp(-1j*alpha_restrictions_2[i][0]), weights, beta_restrictions),
                            method='L-BFGS-B', 
                            bounds=[(0, # alphamin - alphamin
                                    (alpha_restrictions_2[i][1] - alpha_restrictions_2[i][0]) % (2*np.pi)), # alphamax - alphamin
                                    ((1-omega_max_opt)/(1+omega_max_opt), 
                                      (1-omega_min_opt)/(1+omega_min_opt))],
                            tol=1e-4, options={'disp': False})
                        best_a_tmp = res.x[1]*np.exp(1j*res.x[0])*np.exp(1j*alpha_restrictions_2[i][0]) # alpha2 + alphamin
                        
                    a_parameters_tmp = a_parameters
                    a_parameters_tmp[k] = best_a_tmp
                    sum_RSS = project_betas_2(analytic_data_matrix.real, weights, time_points, a_parameters_tmp, beta_restrictions)
                    
                    if sum_RSS < best_sum_RSS:
                        best_sum_RSS = sum_RSS
                        best_a = best_a_tmp
                        
                a_parameters[k] = best_a
                coefs_proj = project_betas(analytic_data_matrix.real, time_points, a_parameters, beta_restrictions)       
                std_remainder = analytic_data_matrix - predict(a_parameters, coefs_proj, time_points)
                weights = 1/np.var(std_remainder, axis=1, ddof=1)

    prediction = predict(a_parameters, coefs_proj, time_points)
    AFD2FMM_matrix = transition_matrix(a_parameters)
    phis = np.dot(AFD2FMM_matrix, coefs_proj.T).T
    
    return a_parameters, coefs_proj, phis, prediction


def RSS_grid(data, est, cosTF, sinTF, weights):
    n_ch = est.shape[1]
    return sum([weights[ch]*np.sum((data[ch] - est[0,ch] - est[1,ch]*cosTF - est[2,ch]*sinTF)**2)  for ch in range(n_ch)])

def RSS_grid_restr(data, est, cosTF, sinTF, sigma_mat, weights, beta_min, beta_max):

    # Loop over columns of vDataMatrix
    for i in range(est.shape[1]):
        OLS = est[:, i:(i+1)]
    
        betaOLS = np.arctan2(-OLS[2], OLS[1]) % (2 * np.pi)
        betaOLS2 = (betaOLS - beta_min) % (2 * np.pi)
    
        if betaOLS2 < (beta_max - beta_min):
            # Valid solutions region
            RLS = OLS
        elif betaOLS2 > (3 * np.pi / 2):
            # Project onto R1
            R = np.array([[0, np.tan(beta_min), 1]]).T  # Column vector (3x1)
            RLS = OLS - sigma_mat @ R @ np.linalg.solve(R.T @ sigma_mat @ R, R.T @ OLS)
        elif betaOLS2 < (beta_max - beta_min + np.pi / 2):
            # Project onto R2
            R = np.array([[0, np.tan(beta_max), 1]]).T  # Column vector (3x1)
            RLS = OLS - sigma_mat @ R @ np.linalg.solve(R.T @ sigma_mat @ R, R.T @ OLS)
        else:
            # Project onto the origin
            RLS = np.array([np.mean(data[i]), 0, 0])
        est[:, i] = RLS.ravel()  # Update pars with RLS
    
    
    return RSS_grid(data, est, cosTF, sinTF, weights)

def opt_mobius_fun_restr(arg, data_matrix, time_points, weights, beta_min, beta_max):
    ts = 2*np.arctan(arg[1]*np.tan((time_points[0] - arg[0])/2)) 
    DM = np.column_stack((np.ones(data_matrix.shape[1]), np.cos(ts), np.sin(ts)))
    sigma_mat = np.linalg.inv(DM.T @ DM)
    est = sigma_mat @ DM.T @ data_matrix.T
    #Weighted RSS
    for i in range(est.shape[1]):
        OLS = est[:, i:(i+1)]
    
        betaOLS = np.arctan2(-OLS[2], OLS[1]) % (2 * np.pi)
        betaOLS2 = (betaOLS - beta_min) % (2 * np.pi)
    
        if betaOLS2 < (beta_max - beta_min):
            # Valid solutions region
            RLS = OLS
        elif betaOLS2 > (3 * np.pi / 2):
            # Project onto R1
            R = np.array([[0, np.tan(beta_min), 1]]).T  # Column vector (3x1)
            RLS = OLS - sigma_mat @ R @ np.linalg.solve(R.T @ sigma_mat @ R, R.T @ OLS)
        elif betaOLS2 < (beta_max - beta_min + np.pi / 2):
            # Project onto R2
            R = np.array([[0, np.tan(beta_max), 1]]).T  # Column vector (3x1)
            RLS = OLS - sigma_mat @ R @ np.linalg.solve(R.T @ sigma_mat @ R, R.T @ OLS)
        else:
            # Project onto the origin
            RLS = np.array([np.mean(data_matrix[i]), 0, 0])
        est[:, i] = RLS.ravel()  # Update pars with RLS
    
    return RSS_grid(data_matrix, est, np.cos(ts), np.sin(ts), weights)

#################################################################################

def fit_fmm_k_mob_restr(data_matrix, time_points=None, n_back=None, max_iter=1,
                        alpha_grid=None, omega_grid=None, 
                        weights=None, post_optimize=True, 
                        omega_min = 0.001, omega_max=1, 
                        beta_min = None, beta_max = None):
    
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
    sigma_mats = [np.linalg.inv(DM.T @ DM) for DM in DMs] # inv(X'X)
    precalculations = [np.linalg.inv(DM.T @ DM) @ DM.T for DM in DMs] # inv(X'X) X' 
    
    ## 1 Iteration of the backfitting algorithm: fit k waves
    for k in range(n_back):
        
        # GRID STEP
        estimates = [prec @ remainder.T for prec in precalculations]

        RSS = [RSS_grid(remainder, est, cosTF[j], sinTF[j], weights) for j, est in enumerate(estimates)]
        RSS = [RSS_grid_restr(remainder, est, cosTF[j], sinTF[j], sigma_mats[j], weights, beta_min, beta_max) for j, est in enumerate(estimates)]
        min_index = np.argmin(RSS) 
        
        
        # OPTIMIZATION STEP
        if(post_optimize):
            res = minimize(opt_mobius_fun_restr, x0=(fmm_grid[min_index]), 
                           args=(remainder, time_points, weights, beta_min, beta_max), 
                           method='L-BFGS-B', bounds=[(-2*np.pi, 4*np.pi),
                                                      (omega_min, omega_max)], 
                           tol=1e-4, options={'disp': False})
            best_pars[k] = res.x
        else:
            best_pars[k] = fmm_grid[min_index]
        
        # PREDICTION AND REMAINDER CALCULATIONS
        ts = 2*np.arctan(best_pars[k][1]*np.tan((time_points[0] - best_pars[k][0])/2)) 
        DM = np.column_stack((np.ones(ts.shape[0]), np.cos(ts), np.sin(ts)))
        sigma_mat = np.linalg.inv(DM.T @ DM)
        linears = sigma_mat @ DM.T @ remainder.T
        
        for i in range(n_ch):
            OLS = linears[:, i:(i+1)]
        
            betaOLS = np.arctan2(-OLS[2], OLS[1]) % (2 * np.pi)
            betaOLS2 = (betaOLS - beta_min) % (2 * np.pi)
        
            if betaOLS2 < (beta_max - beta_min):
                # Valid solutions region
                RLS = OLS
            elif betaOLS2 > (3 * np.pi / 2):
                # Project onto R1
                R = np.array([[0, np.tan(beta_min), 1]]).T  # Column vector (3x1)
                RLS = OLS - sigma_mat @ R @ np.linalg.solve(R.T @ sigma_mat @ R, R.T @ OLS)
            elif betaOLS2 < (beta_max - beta_min + np.pi / 2):
                # Project onto R2
                R = np.array([[0, np.tan(beta_max), 1]]).T  # Column vector (3x1)
                RLS = OLS - sigma_mat @ R @ np.linalg.solve(R.T @ sigma_mat @ R, R.T @ OLS)
            else:
                # Project onto the origin
                RLS = np.array([np.mean(remainder[i]), 0, 0])
            linears[:, i] = RLS.ravel()
        
        components[k] = np.column_stack([linears[0,ch] + linears[1,ch]*np.cos(ts) + linears[2,ch]*np.sin(ts) for ch in range(n_ch)]).T
        remainder = remainder - components[k]
        weights = 1/np.var(remainder, axis = 1)
        
        best_pars_linear[k] = linears
        
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
                    res = minimize(opt_mobius_fun_restr, x0=(fmm_grid[min_index]), 
                                   args=(remainder, time_points, weights, beta_min, beta_max), 
                                   method='L-BFGS-B', bounds=[(-2*np.pi, 4*np.pi),
                                                              (omega_min, omega_max)], 
                                   tol=1e-4, options={'disp': False})
                    best_pars[k] = res.x
                else:
                    best_pars[k] = fmm_grid[min_index]
                
                # PREDICTION AND REMAINDER CALCULATIONS
                ts = 2*np.arctan(best_pars[k][1]*np.tan((time_points[0] - best_pars[k][0])/2)) 
                DM = np.column_stack((np.ones(ts.shape[0]), np.cos(ts), np.sin(ts)))
                sigma_mat = np.linalg.inv(DM.T @ DM)
                linears = sigma_mat @ DM.T @ remainder.T
    
                for i in range(n_ch):
                    OLS = linears[:, i:(i+1)]
                
                    betaOLS = np.arctan2(-OLS[2], OLS[1]) % (2 * np.pi)
                    betaOLS2 = (betaOLS - beta_min) % (2 * np.pi)
                
                    if betaOLS2 < (beta_max - beta_min):
                        # Valid solutions region
                        RLS = OLS
                    elif betaOLS2 > (3 * np.pi / 2):
                        # Project onto R1
                        R = np.array([[0, np.tan(beta_min), 1]]).T  # Column vector (3x1)
                        RLS = OLS - sigma_mat @ R @ np.linalg.solve(R.T @ sigma_mat @ R, R.T @ OLS)
                    elif betaOLS2 < (beta_max - beta_min + np.pi / 2):
                        # Project onto R2
                        R = np.array([[0, np.tan(beta_max), 1]]).T  # Column vector (3x1)
                        RLS = OLS - sigma_mat @ R @ np.linalg.solve(R.T @ sigma_mat @ R, R.T @ OLS)
                    else:
                        # Project onto the origin
                        RLS = np.array([np.mean(remainder[i]), 0, 0])
                    linears[:, i] = RLS.ravel()
                    
                components[k] = np.column_stack([linears[0,ch] + linears[1,ch]*np.cos(ts) + linears[2,ch]*np.sin(ts) for ch in range(n_ch)]).T
                remainder = remainder - components[k]
                weights = 1/np.var(remainder, axis = 1)    
                best_pars_linear[k] = linears
            
    return best_pars, best_pars_linear, remainder
    
