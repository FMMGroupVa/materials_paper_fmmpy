# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 10:07:03 2025

@author: Christian
"""

import numpy as np
from numpy.fft import fft, ifft
from scipy.optimize import minimize, Bounds
from numba import jit
import matplotlib.pyplot as plt
from auxiliar_functions import seq_times, szego, mobius, predict, predict2, transition_matrix, inner_products_sum_2, split_complex
from qpsolvers import solve_ls, solve_qp

def inner_products_sum_restr(splitted_a, analytic_data_matrix, time_points, k, a_parameters, weights, beta_min, beta_max):
    
    a_parameters[k] = splitted_a[1]*np.exp(1j*splitted_a[0])
    coefs = project_betas(analytic_data_matrix.real, time_points, a_parameters, beta_min, beta_max)
    
    # return -np.sum([weights[ch_i]*(np.abs(coefs_r) ** 2) for ch_i, coefs_r in enumerate(coefs)])
    return -np.sum(weights[:, np.newaxis] * np.abs(coefs) ** 2)

# Generates a matrix with p restrictions (between a and b)
def generate_G(p, a, b):
    G = np.zeros((2*p, 2*p+1))
    for var in range(p):
        G[2*var, 2*var+1] = np.sin(a)
        G[2*var, 2*var+2] = np.cos(a)
        G[2*var+1, 2*var+1] = -np.sin(b)
        G[2*var+1, 2*var+2] = -np.cos(b)
    return G

def project_betas(data_matrix, time_points, a, beta_min, beta_max):
    n_back = len(a) - 1
    n_ch, n_obs = data_matrix.shape
    
    # 1. AFD to Complex FMM
    G = generate_G(n_back, beta_min, beta_max)
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
        # RLS[ch_i] = solve_ls(DM, data_matrix[ch_i], G=G, h=h, solver='quadprog')
        RLS[ch_i] = solve_qp(DM.T@DM, -DM.T@data_matrix[ch_i], G=G, h=h, solver='quadprog')
    # RLS = np.apply_along_axis(lambda row: solve_ls(DM, row, G=G, h=h, solver='quadprog'), axis=1, arr=data_matrix)
    
    # 6. Compute betas, amplitudes, and phis using vectorized operations
    betas = np.arctan2(-RLS[:, 2::2], RLS[:, 1::2])
    amplitudes = np.sqrt(RLS[:, 1::2] ** 2 + RLS[:, 2::2] ** 2)
    phis[:, 0] = RLS[:, 0]
    phis[:, 1:] = amplitudes * np.exp(1j * (betas - np.angle(a[1:]) + np.pi))

    # Return AFD coefs
    return np.dot(np.linalg.inv(transition_matrix(a)), phis.T).T

def fit_fmm_k_restr(analytic_data_matrix, time_points=None, n_back=None, max_iter=None,
                    omega_grid=None, weights=None, post_optimize=True, 
                    omega_min = 0.001, omega_max=0.999, 
                    alpha_restrictions=None, omega_restrictions=None):
    
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
    for k in range(1, n_back+1):
        
        basek = base[(omega_grid>omega_restrictions[k-1][0]) & (omega_grid<omega_restrictions[k-1][1])]
        an_search_len_restr = basek.shape[0]
        ## STEP 1: Grid search - AFD-FFT formulations
        abs_coefs = 0
        for ch_i in range(n_ch):
            #abs_coefs += np.abs(ifft(pymat.repmat(fft(analytic_data_matrix[ch_i, :], n_obs), an_search_len, 1) * base, n_obs, 1))
            abs_coefs += weights[ch_i]*np.abs(ifft(np.repeat(fft(
                remainder[ch_i, :], n_obs)[np.newaxis, :], 
                an_search_len_restr, axis=0) * basek, n_obs, 1))
        abs_coefs = abs_coefs.T # n_obs x n_omegas
        
        # Best a: we only select alphas in the restricted arc
        afd_grid2 = afd_grid[:,(omega_grid>omega_restrictions[k-1][0]) & (omega_grid<omega_restrictions[k-1][1])]
        if(alpha_restrictions_2[k-1][0] > alpha_restrictions_2[k-1][1]):
            # +++]--------[+++
            abs_coefs = abs_coefs[(time_points[0]>=alpha_restrictions_2[k-1][0]) | (time_points[0]<=alpha_restrictions_2[k-1][1]) ]
            afd_grid2 = afd_grid2[(time_points[0]>=alpha_restrictions_2[k-1][0]) | (time_points[0]<=alpha_restrictions_2[k-1][1]) ]
        else:
            # ------[++++]----
            abs_coefs = abs_coefs[(time_points[0]>=alpha_restrictions_2[k-1][0]) & (time_points[0]<=alpha_restrictions_2[k-1][1]) ]
            afd_grid2 = afd_grid2[(time_points[0]>=alpha_restrictions_2[k-1][0]) & (time_points[0]<=alpha_restrictions_2[k-1][1]) ]
        
        max_loc_tmp = np.argwhere(abs_coefs == np.amax(abs_coefs))
        best_a = afd_grid2[max_loc_tmp[0, 0], max_loc_tmp[0, 1]]


        ## STEP 2: Postoptimization - Profile log-likelihood.
        if(post_optimize):
            # We transform time points as: ---[+++]-----  ->  [+++]--------
            time_points_transformed = time_points - alpha_restrictions[k-1][0] 
            # Lower values than the general omega_min are not allowed
            omega_min_opt = max(omega_min, omega_restrictions[k-1][0])
            omega_max_opt = min(omega_max, omega_restrictions[k-1][1])
            # Optimization routine
            res = minimize(
                inner_products_sum_2, x0=split_complex(best_a), 
                args=(remainder, time_points_transformed, weights), 
                method='L-BFGS-B', 
                bounds=[(0, # alphamin - alphamin
                        (alpha_restrictions_2[k-1][1] - alpha_restrictions_2[k-1][0]) % (2*np.pi)), # alphamax - alphamin
                        ((1-omega_max_opt)/(1+omega_max_opt), 
                         (1-omega_min_opt)/(1+omega_min_opt))],
                tol=1e-4, options={'disp': False})
            opt_a = res.x[1]*np.exp(1j*res.x[0])
            a_parameters[k] = opt_a * np.exp(1j*alpha_restrictions[k-1][0]) # alpha + alphamin
        else:
            a_parameters[k] = best_a
        
        # Coefficient calculations 
        szego_a = szego(a_parameters[k], time_points)
        for ch_i in range(n_ch):
            coefs[ch_i, k] = np.conj(szego_a.dot(remainder[ch_i,:].conj().T))/n_obs
            remainder[ch_i,:] = ((remainder[ch_i,:] - coefs[ch_i,k]*szego_a) 
                                 / mobius(a_parameters[k], time_points))
        
        AFD2FMM_matrix = transition_matrix(a_parameters[0:k+1])
        for ch_i in range(n_ch):    
            phis[ch_i, 0:k+1] = np.dot(AFD2FMM_matrix, coefs[ch_i, 0:k+1].T).T
    
    if max_iter > 1:
        for iter_j in range(1, max_iter):
            # Auxiliar Blaschke product: z*Bl_{a_1,...,a_K} = z*m(a1,t)*...*m(aK,t)
            blaschke = z 
            for k in range(1, n_back+1):
                blaschke = blaschke*mobius(a_parameters[k], time_points)
        
            for k in range(1, n_back+1):
                # Calculate the standard reminder (data-prediction) without component k:  r = X - sum ci*Bi, i != k
                # std_remainder = analytic_data_matrix - predict(np.delete(a_parameters, k, axis=0), np.delete(coefs, k, axis=1), time_points)
                std_remainder = analytic_data_matrix - predict2(np.delete(a_parameters, k, axis=0), analytic_data_matrix, time_points)[0]
                
                # weights = data_norm - np.sum(np.abs(np.delete(coefs, [0, k], axis=1))**2, axis=1)/2
                weights = 1/np.var(std_remainder.real, axis=1)
                
                # Calculate the reduced reminder reminder/(z*mob1*...,mobK) (without k)
                blaschke = blaschke / mobius(a_parameters[k], time_points)
                # Reduced reminder (to calculate )
                remainder = std_remainder / blaschke
                
                # Select basis components whose abs(a) / omegas are in the restricted range
                basek = base[(omega_grid>omega_restrictions[k-1][0]) & (omega_grid<omega_restrictions[k-1][1])]
                an_search_len_restr = basek.shape[0]
                ## STEP 1: Grid search - AFD-FFT formulations
                abs_coefs = 0
                for ch_i in range(n_ch):
                    #abs_coefs += np.abs(ifft(pymat.repmat(fft(analytic_data_matrix[ch_i, :], n_obs), an_search_len, 1) * base, n_obs, 1))
                    abs_coefs += weights[ch_i]*np.abs(ifft(np.repeat(fft(
                        remainder[ch_i, :], n_obs)[np.newaxis, :], 
                        an_search_len_restr, axis=0) * basek, n_obs, 1))
                abs_coefs = abs_coefs.T
                
                # Best a: we only select alphas in the restricted arc
                afd_grid2 = afd_grid[:,(omega_grid>omega_restrictions[k-1][0]) & (omega_grid<omega_restrictions[k-1][1])]
                if(alpha_restrictions_2[k-1][0] > alpha_restrictions_2[k-1][1]):
                    # +++]--------[+++  -> restricted alphas arc crossing 0 alphaMin > alphaMax
                    abs_coefs = abs_coefs[(time_points[0]>=alpha_restrictions_2[k-1][0]) | (time_points[0]<=alpha_restrictions_2[k-1][1])]
                    afd_grid2 = afd_grid2[(time_points[0]>=alpha_restrictions_2[k-1][0]) | (time_points[0]<=alpha_restrictions_2[k-1][1])]
                else:
                    # ------[++++]----  -> restricted alphas in [alphaMin, alphaMax] alphaMin < alphaMax
                    abs_coefs = abs_coefs[(time_points[0]>=alpha_restrictions_2[k-1][0]) & (time_points[0]<=alpha_restrictions_2[k-1][1])]
                    afd_grid2 = afd_grid2[(time_points[0]>=alpha_restrictions_2[k-1][0]) & (time_points[0]<=alpha_restrictions_2[k-1][1])]
                    
                max_loc_tmp = np.argwhere(abs_coefs == np.amax(abs_coefs))
                best_a = afd_grid2[max_loc_tmp[0, 0], max_loc_tmp[0, 1]]
                
                ## STEP 2: Postoptimization - Profile log-likelihood.
                if(post_optimize):
                    # We transform time points as: ---[+++]-----  ->  [+++]--------
                    time_points_transformed = time_points - alpha_restrictions_2[k-1][0] 
                    # Lower values than the general omega_min are not allowed
                    omega_min_opt = max(omega_min, omega_restrictions[k-1][0])
                    omega_max_opt = min(omega_max, omega_restrictions[k-1][1])
                    # Optimization routine
                    res = minimize(
                        inner_products_sum_2, x0=split_complex(best_a), 
                        args=(remainder, time_points_transformed, weights), 
                        method='L-BFGS-B',
                        bounds=[(0, # alphamin - alphamin
                                (alpha_restrictions_2[k-1][1] - alpha_restrictions_2[k-1][0]) % (2*np.pi)), # alphamax - alphamin
                                ((1-omega_max_opt)/(1+omega_max_opt), 
                                 (1-omega_min_opt)/(1+omega_min_opt))],
                        tol=1e-4, options={'disp': False})
                    opt_a = res.x[1]*np.exp(1j*res.x[0])
                    a_parameters[k] = opt_a * np.exp(1j*alpha_restrictions_2[k-1][0]) # alpha + alphamin
                else:
                    a_parameters[k] = best_a
                
                # Coefficient calculations c_k = <Remainder, szego(a)>
                szego_a = szego(a_parameters[k], time_points)
                for ch_i in range(n_ch):
                    coefs[ch_i, k] = np.conj(szego_a.dot(remainder[ch_i,:].conj().T))/n_obs
                
                # Blaschke product update
                blaschke = blaschke * mobius(a_parameters[k], time_points)
                
    AFD2FMM_matrix = transition_matrix(a_parameters)
    phis= np.dot(AFD2FMM_matrix, coefs.T).T
    
    prediction, coefs = predict2(a_parameters, analytic_data_matrix, time_points)
    
    return a_parameters, coefs, phis, prediction


def fit_fmm_k_restr_betas(analytic_data_matrix, time_points=None, n_back=None, max_iter=None,
              omega_grid=None, weights=None, post_optimize=True, omega_min=0.001, omega_max=0.99, 
              beta_min=None, beta_max=None):
    
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
    coefs2 = np.zeros((n_ch, n_back+1), dtype=complex) # Auxiliar for restr
    # phis2 = np.zeros((n_ch, n_back+1), dtype=complex) # Auxiliar for restr
    a_parameters = np.zeros(n_back+1, dtype=complex)

    # Start decomposing: c0 (mean)
    # Remainder R_(k+1) = ( R_k - c_k*e_ak(t) ) / m_ak(t)
    # e_ak := szego(ak, t)
    # m_ak := mobius(ak, t)
    z = np.exp(1j*time_points)
    remainder = np.copy(analytic_data_matrix)
    for ch_i in range(n_ch):
        coefs[ch_i,0] = np.mean(analytic_data_matrix[ch_i,:])
        coefs2[ch_i,0] = np.mean(analytic_data_matrix[ch_i,:])
        remainder[ch_i,:] = ((analytic_data_matrix[ch_i,:] - coefs[ch_i,0])/z)
    # transition_mat = np.zeros((n_back+1, n_back+1), dtype = "complex")
    # coefs_grid = [None] * n_ch
    weights = 1/np.var(remainder.real, axis=1, ddof=1)
    
    ## 1 Iteration of the backfitting algorithm: fit k waves
    for k in range(1, n_back+1):
        
        ## STEP 1: Grid search - AFD-FFT formulations
        abs_coefs = 0
        
        for ch_i in range(n_ch):
            abs_coefs += weights[ch_i]*np.abs(ifft(np.repeat(fft(
                remainder[ch_i, :], n_obs)[np.newaxis, :], 
                an_search_len, axis=0) * base, n_obs, 1))**2
        
        abs_coefs = abs_coefs.T
        
        #######################################################################
        
        candidate_as = afd_grid[::4,:]  
        
        abs_coefs_2 = abs_coefs[::4,:] 

        for index, value in enumerate(candidate_as.ravel()):
            i, j = np.unravel_index(index, candidate_as.shape)
            if not np.isin(value, a_parameters):
                a_parameters[k] = value
                coefs = project_betas(analytic_data_matrix.real, time_points, a_parameters[:(k+1)], beta_min, beta_max)
                abs_coefs_2[i,j] = np.sum([weights[ch_i] * (np.abs(coefs_r) ** 2) for ch_i, coefs_r in enumerate(coefs)])
            else:
                abs_coefs_2[i,j] = 0
        
        max_loc_tmp = np.argwhere(abs_coefs_2 == np.amax(abs_coefs_2))
        best_a = candidate_as[max_loc_tmp[0, 0], max_loc_tmp[0, 1]]
        a_parameters[k] = best_a
        
        if(post_optimize):
            res = minimize(
                inner_products_sum_restr, x0=split_complex(best_a), 
                args=(analytic_data_matrix, time_points, k, a_parameters[:(k+1)], weights, beta_min, beta_max), 
                # Bounds: (-2pi, 4pi) para explorar bien parametro circular
                method='L-BFGS-B', bounds=[(-2*np.pi, 4*np.pi), ((1-omega_max)/(1+omega_max), (1-omega_min)/(1+omega_min))],
                tol=1e-4, options={'disp': False})
            opt_a = res.x[1]*np.exp(1j*res.x[0])
            a_parameters[k] = opt_a

        # Coefficient calculations 
        szego_a = szego(a_parameters[k], time_points)
        
        for ch_i in range(n_ch):
            coefs[ch_i, k] = np.conj(szego_a.dot(remainder[ch_i,:].conj().T))/n_obs
            remainder[ch_i,:] = ((remainder[ch_i,:] - coefs[ch_i,k]*szego_a) / mobius(a_parameters[k], time_points))
        
        coefs_proj = project_betas(analytic_data_matrix.real, time_points, a_parameters[:(k+1)], beta_min, beta_max)



    if max_iter > 1:
        for iter_j in range(1,max_iter):
            # Auxiliar Blaschke product: z*Bl_{a_1,...,a_K} = z*m(a1,t)*...*m(aK,t)
            blaschke = z 
            for k in range(1, n_back+1):
                blaschke = blaschke*mobius(a_parameters[k], time_points)
        
            for k in range(1, n_back+1):
                # Calculate the standard reminder (data-prediction) without component k:  r = X - sum ci*Bi, i != j
                # std_remainder = analytic_data_matrix - predict(np.delete(a_parameters, k, axis=0), np.delete(coefs, k, axis=1), time_points)
                coefs_proj = project_betas(analytic_data_matrix.real, time_points, np.delete(a_parameters, k, axis=0), beta_min, beta_max)                
                std_remainder = analytic_data_matrix - predict(np.delete(a_parameters, k, axis=0), coefs_proj, time_points)
                weights = 1/np.var(std_remainder, axis=1, ddof=1)
                
                # Calculate the reduced reminder reminder/(z*mob1*...,mobK) (without k)
                blaschke = blaschke / mobius(a_parameters[k], time_points)
                remainder = std_remainder / blaschke
                
                abs_coefs = 0
                
                for ch_i in range(n_ch):
                    abs_coefs += weights[ch_i]*np.abs(
                        ifft(np.repeat(fft(remainder[ch_i, :], n_obs)[np.newaxis, :], an_search_len, axis=0) * base, n_obs, 1)
                        )**2
                
                abs_coefs = abs_coefs.T
                
                
                #######################################################################
                candidate_as = afd_grid[::4,:]  
                abs_coefs_2 = abs_coefs[::4,:] 

                for index, value in enumerate(candidate_as.ravel()):
                    i, j = np.unravel_index(index, candidate_as.shape)
                    if(not np.isin(value, a_parameters)):
                        a_parameters[k] = value
                        coefs = project_betas(analytic_data_matrix.real, time_points, a_parameters[:(k+1)], beta_min, beta_max)
                        abs_coefs_2[i,j] = np.sum([weights[ch_i] * (np.abs(coefs_r) ** 2) for ch_i, coefs_r in enumerate(coefs)])
                    else:
                        abs_coefs_2[i,j] = 0
                
            
                max_loc_tmp = np.argwhere(abs_coefs_2 == np.amax(abs_coefs_2))
                best_a = candidate_as[max_loc_tmp[0, 0], max_loc_tmp[0, 1]]
                a_parameters[k] = best_a
                
                szego_a = szego(a_parameters[k], time_points)
                for ch_i in range(n_ch):
                    coefs2[ch_i, k] = np.conj(szego_a.dot(remainder[ch_i,:].conj().T))/n_obs
                    # betas[ch_i, 0:k] = (np.angle(phis2[ch_i, 1:(k+1)]) + np.angle(a_parameters[1:(k+1)]) - np.pi)%(2*np.pi)
                    remainder[ch_i,:] = ((remainder[ch_i,:] - coefs[ch_i,k]*szego_a) / mobius(a_parameters[k], time_points))
                coefs_proj = project_betas(analytic_data_matrix.real, time_points, a_parameters, beta_min, beta_max)
                
                abs_coefs = 0
                for ch_i in range(n_ch):
                    abs_coefs += weights[ch_i]*np.abs(coefs_proj[ch_i,:])**2
                
                
                # STEP 2: Postoptimization - Profile log-likelihood.
                if(post_optimize):
                    res = minimize(
                        inner_products_sum_restr, x0=split_complex(best_a), 
                        args=(analytic_data_matrix, time_points, k, a_parameters, coefs, weights, beta_min, beta_max), 
                        # Bounds: (-2pi, 4pi) para explorar bien parametro circular
                        method='L-BFGS-B', bounds=[(-2*np.pi, 4*np.pi), ((1-omega_max)/(1+omega_max), (1-omega_min)/(1+omega_min))],
                        tol=1e-4, options={'disp': False})
                    opt_a = res.x[1]*np.exp(1j*res.x[0])
                    a_parameters[k] = opt_a
                
                
                # Coefficient calculations 
                szego_a = szego(a_parameters[k], time_points)
                for ch_i in range(n_ch):
                    coefs[ch_i, k] = np.conj(szego_a.dot(remainder[ch_i,:].conj().T))/n_obs
                    remainder[ch_i,:] = ((remainder[ch_i,:] - coefs[ch_i,k]*szego_a) 
                                          / mobius(a_parameters[k], time_points))
                
                coefs_proj = project_betas(analytic_data_matrix.real, time_points, a_parameters, beta_min, beta_max)

                blaschke = blaschke * mobius(a_parameters[k], time_points)
    
    prediction = predict(a_parameters, coefs_proj, time_points)
    
    AFD2FMM_matrix = transition_matrix(a_parameters)
    prediction, coefs = predict2(a_parameters, analytic_data_matrix, time_points)
    phis = np.dot(AFD2FMM_matrix, coefs.T).T
    
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
    