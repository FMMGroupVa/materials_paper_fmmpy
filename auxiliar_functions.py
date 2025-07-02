# -*- coding: utf-8 -*-

import numpy as np
from numba import jit

@jit
def szego(a, t): 
    return ((1 - np.abs(a)**2) ** 0.5) / (1 - np.conj(a)*np.exp(1j*t))

@jit
def mobius(a, t): 
    return ((np.exp(1j*t) - a)) / (1 - np.conj(a)*np.exp(1j*t))

@jit
def split_complex(z): 
    return ((np.angle(z) % (2*np.pi), np.abs(z)))

def seq_times(nObs):
    return np.reshape(np.linspace(0, 2 * np.pi, num=nObs+1)[:-1], (1, nObs))

def predict(a, coefs, time_points):
    n_ch, n_coefs = coefs.shape
    n_obs = time_points.shape[1]

    prediction = np.ones((n_ch, n_obs), dtype = complex)*coefs[:,0:1]
    blaschke = np.ones((1, n_obs))

    for k in range(1, n_coefs):
        for ch_i in range(n_ch):
            prediction[ch_i] = prediction[ch_i] + coefs[ch_i,k]*np.exp(1j*time_points)*szego(a[k], time_points)*blaschke
        blaschke = blaschke*mobius(a[k], time_points)
    return prediction

# Versión de predict donde los a's están fijos y los coefs se recalculan 
def predict2(a, analytic_data_matrix, time_points):

    n_back = a.shape[0]-1
    if(analytic_data_matrix.ndim == 2):
        n_ch, n_obs = analytic_data_matrix.shape
    else:
        n_obs = analytic_data_matrix.shape[0]
        n_ch = 1
        analytic_data_matrix = analytic_data_matrix[np.newaxis, :]

    # Hay que recalcular todos los coeficientes
    coefs = np.zeros((n_ch, n_back+1), dtype=complex)
    
    z = np.exp(1j*time_points)
    remainder = np.copy(analytic_data_matrix)
    
    # Primero el coeficiente 0
    for ch_i in range(n_ch):
        coefs[ch_i,0] = np.mean(analytic_data_matrix[ch_i,:])
        remainder[ch_i,:] = ((analytic_data_matrix[ch_i,:] - coefs[ch_i,0])/z)
        
    # Ahora coeficientes 1,...,K
    for k in range(1, n_back+1):
        szego_a = szego(a[k], time_points) # El a está fijo aquí
        for ch_i in range(n_ch):
            coefs[ch_i, k] = np.conj(szego_a.dot(remainder[ch_i,:].conj().T))/n_obs
            remainder[ch_i,:] = ((remainder[ch_i,:] - coefs[ch_i,k]*szego_a) 
                                 / mobius(a[k], time_points))

    return predict(a, coefs, time_points), coefs

def predictFMM(a, phis, time_points):

    n_ch, n_coefs = phis.shape
    n_obs = time_points.shape[1]

    prediction = np.ones((n_ch, n_obs), dtype = complex) * phis[:,0][:, np.newaxis]

    for k in range(1, n_coefs):
        for ch_i in range(n_ch):
            prediction[ch_i] = prediction[ch_i] + phis[ch_i, k]*mobius(a[k], time_points)
    return prediction

###############################################################################

# Change of basis FMM <=> AFD codes

# m1(t)*m2(t) = beta0 + beta1*m1(t) + beta2*m2(t)
def beta0(a1, a2):
    return (a1 - a2) / (np.conj(a1) - np.conj(a2))

def beta1(a1, a2):
    return (1 - np.conj(a1) * a2) / (np.conj(a1) - np.conj(a2))

# Row k containts al pairs mi, mj for i,j=1,...K
def beta_matrix(an):
    N = len(an)
    beta0Mat = np.zeros((N, N), dtype=complex)
    beta1Mat = np.zeros((N, N), dtype=complex)
    
    for i in range(N-1):
        for j in range(i+1, N):
            beta0Mat[i, j] = beta0(an[i], an[j])
            beta1Mat[i, j] = beta1(an[i], an[j])
            beta1Mat[j, i] = beta1(an[j], an[i])  # Beta2(i,j) = Beta1(j,i)
    
    beta0Mat = beta0Mat + beta0Mat.T
    return beta0Mat, beta1Mat

# matrix built row-by-row -> m1-> {m1, m2} -> ... -> {m1,m2,...,mK}
# m1(t)*m2(t)*...*mk(t) = beta0 + beta1*m1(t) + beta2*m2(t) + ... + betak*mk(t)
def calculate_xi_matrix(an):
    beta0_mat, beta1_mat = beta_matrix(an)
    N = len(an)
    
    # Initialize xi_mat with an extra row of zeros at the end (size (N+1, N+1))
    xi_mat = np.zeros((N + 1, N + 1), dtype=complex)
    
    xi_mat[0, 0] = 1
    xi_mat[1, 1] = 1
    # Loop through k from 2 to N
    for k in range(2, N+1):
        # Vector prev_xis: [xi_1(k-1), ..., xi_k-1(k-1)] (indexing starts at 0)
        prev_xis = xi_mat[k-1, 1:k]

        # Compute xi_0(k) as the dot product of prev_xis and the corresponding beta0_mat column
        xi_mat[k, 0] = np.dot(prev_xis, beta0_mat[:k-1, k-1])
        
        # Compute xi_1(k), ..., xi_k-1(k) using element-wise multiplication
        xi_mat[k, 1:k] = beta1_mat[:k-1, k-1] * prev_xis
        
        # Compute xi_k(k) as a scalar product of [xi0(k-1), prev_xis] and [1, beta1(1,k), ..., beta1(k-1,k)]
        xi_mat[k, k] = xi_mat[k-1, 0] + np.dot(prev_xis, beta1_mat[k-1, :k-1])
    
    return xi_mat

# Change of basis matrix FMM <=> AFD
def transition_matrix(an):
    """
    Computes the transition matrix M for the change of basis from B to B' where:
    B' is a Möbius basis and B is a Takenaka-Malmquist basis.
    
    Parameters:
    an : numpy array
        A vector of size [K] with parameters a1,...,aK all different.
        
    Returns:
    M : numpy array
        Transition matrix of dimension (K+1)x(K+1).
    """
    K = len(an)-1  # Length of the an vector
    M = np.zeros((K+1, K+1), dtype=complex)  # Initialize the (K+1)x(K+1) matrix with zeros
    xiMat = calculate_xi_matrix(an[1:])  # Call the xiMatrix function

    # The 1st column 1 is (1,0,...,0)': both bases have an intercept as first element
    M[0, 0] = 1

    # Loop through each k for the first column and the upper triangular part
    for k in range(K):
        # c0 coefficient (first row, element by element)
        M[0, k+1] = (an[k+1] * xiMat[k, 0] + xiMat[k+1, 0]) / np.sqrt(1 - np.abs(an[k+1])**2)
        
        # Triangular matrix: only upper triangle is not zero. Each k is a row.
        for j in range(k+1, K):
            
            M[k+1, j+1] = (np.sqrt(1 - np.abs(an[j+1])**2) * xiMat[j, k+1] /
                           (np.conj(an[k+1]) - np.conj(an[j+1])))

        # Diagonal elements of M
        M[k+1, k+1] = xiMat[k+1, k+1] / np.sqrt(1 - np.abs(an[k+1])**2)

    return M

###############################################################################

def inner_products_sum(splitted_a, analytic_data_matrix, t, weights):
    a = splitted_a[1]*np.exp(1j*splitted_a[0])
    sum_abs = 0
    for ch_i in range(analytic_data_matrix.shape[0]):
        sum_abs = sum_abs + weights[ch_i]*(np.abs(np.conj(szego(a, t).dot(analytic_data_matrix[ch_i,:].conj().T))) ** 2)
    return -np.sum(sum_abs)

# Pensaba que esta version podría ser un poco más rápida que la anterior,
# no hay diferencias
def inner_products_sum_2(splitted_a, analytic_data_matrix, t, weights):

    a = splitted_a[1]*np.exp(1j*splitted_a[0])
    return -sum([weights[ch_i]*(np.abs(np.conj(szego(a, t).dot(analytic_data_matrix[ch_i,:].conj().T))) ** 2) 
                 for ch_i in range(analytic_data_matrix.shape[0])])


###############################################################################

# Mobius functions -> To do software tests (Private)

def _opt_mobius_fun(arg, data_matrix, time_points, weights):
    ts = 2*np.arctan(arg[1]*np.tan((time_points[0] - arg[0])/2)) 
    DM = np.column_stack((np.ones(data_matrix.shape[1]), np.cos(ts), np.sin(ts)))
    linears = np.linalg.inv(DM.T @ DM) @ DM.T @ data_matrix.T
    #Weighted RSS
    return(np.sum([weights[ch]*np.sum((data_matrix[ch] - linears[0,ch] - linears[1,ch]*np.cos(ts) - linears[2,ch]*np.sin(ts))**2) for ch in range(data_matrix.shape[0])]))


def _fit_fmm_k_mob(data_matrix, time_points=None, n_back=None, max_iter=1,
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
            res = minimize(_opt_mobius_fun, x0=(fmm_grid[min_index]), 
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
                    res = minimize(_opt_mobius_fun, x0=(fmm_grid[min_index]), 
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
    

def _RSS_grid(data, est, cosTF, sinTF, weights):
    n_ch = est.shape[1]
    return sum([weights[ch]*np.sum((data[ch] - est[0,ch] - est[1,ch]*cosTF - est[2,ch]*sinTF)**2)  for ch in range(n_ch)])


