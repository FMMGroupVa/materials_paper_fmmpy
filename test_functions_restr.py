# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 09:13:49 2025

@author: Christian
"""


import numpy as np
import pandas as pd
import scipy.signal as sc
import matplotlib.pyplot as plt
import os
os.chdir(r'C:\Users\Christian\Documents\GitHub\PaquetePython')
import time
# from  fit_fmm_unit import fit_fmm_unit
from auxiliar_functions import seq_times


from fit_fmm_k import fit_fmm_k, fit_fmm_k_mob
from auxiliar_functions import predict, predict2, predictFMM, seq_times, transition_matrix
from fit_fmm_k_restr import fit_fmm_k_restr, fit_fmm_k_mob_restr, fit_fmm_k_restr_betas, project_betas

df2 = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\Chroma_example.csv').T
df2 = df2.iloc[0:10,:]
time_points2 = seq_times(df2.shape[1])
analytic_data_matrix2 = sc.hilbert(df2, axis = 1)
n_ch2, n_obs2 = analytic_data_matrix2.shape

omega_grid = np.linspace(0.001, 0.1, 50, False)

#%%
n_back = 30
max_iter = 1
a, coefs, phis, prediction = fit_fmm_k(
    analytic_data_matrix=analytic_data_matrix2, n_back=n_back, max_iter=max_iter,
    time_points=time_points2, omega_grid=omega_grid,
    weights=np.ones(n_ch2), post_optimize=False, 
    omega_max = 0.999)

ch = 0
plt.plot(time_points2[0], analytic_data_matrix2[ch].real, color='blue')
plt.plot(time_points2[0], prediction[ch], color='purple')
plt.show()

#%%

alpha_grid = seq_times(50)
omega_grid = np.linspace(0.001, 0.5, 50, False)
n_back = 5
max_iter = 2
best_pars, best_pars_linear, remainder = fit_fmm_k_mob(data_matrix=analytic_data_matrix2.real,  n_back=n_back, max_iter=max_iter, 
                                                   time_points=time_points2, omega_grid=omega_grid, alpha_grid=alpha_grid, 
                                                   weights=np.ones(n_ch2))


#%%

for k in range(n_back):
    alpha, omega = best_pars[k]
    ts = 2*np.arctan(omega*np.tan((time_points2[0] - alpha)/2)) 
    delta, gamma = best_pars_linear[k][1:3,ch]
    comp = delta*np.cos(ts) + gamma*np.sin(ts)
    plt.plot(time_points2[0], comp-comp[0], color='purple')
    
plt.show()

#%%

alpha_grid = seq_times(50)
omega_grid = np.linspace(0.001, 0.5, 50, False)
n_back = 10
max_iter = 5
inicio = time.time()
best_pars, best_pars_linear, remainder = fit_fmm_k_mob_restr(data_matrix=analytic_data_matrix2.real,  n_back=n_back, max_iter=max_iter, 
                                                   time_points=time_points2, omega_grid=omega_grid, alpha_grid=alpha_grid, 
                                                   weights=np.ones(n_ch2), beta_min=np.pi-0.5, beta_max=np.pi+0.5)
fin = time.time()
print(fin-inicio)

#%%
ch = 0
plt.plot(time_points2[0], analytic_data_matrix2[ch].real, color='blue')
plt.plot(time_points2[0], analytic_data_matrix2[ch].real - remainder[ch], color='purple')
plt.show()

#%%

for k in range(n_back):
    alpha, omega = best_pars[k]
    ts = 2*np.arctan(omega*np.tan((time_points2[0] - alpha)/2)) 
    delta, gamma = best_pars_linear[k][1:3,ch]
    comp = delta*np.cos(ts) + gamma*np.sin(ts)
    plt.plot(time_points2[0], comp-comp[0], color='purple')
    
plt.show()

#%%
from fit_fmm_k_restr import fit_fmm_k_restr, fit_fmm_k_mob_restr, fit_fmm_k_restr_betas, project_betas

n_back = 1
max_iter = 1
a, coefs, phis, prediction2 = fit_fmm_k_restr_betas(
    analytic_data_matrix=analytic_data_matrix2, n_back=n_back, max_iter=max_iter,
    time_points=time_points2, omega_grid=omega_grid,
    weights=np.ones(n_ch2), post_optimize=True, 
    beta_min=np.pi-0.5, beta_max=np.pi+0.5)

#%%
plt.plot(time_points2[0], analytic_data_matrix2[0].real, color='blue')
#plt.plot(time_points[0], prediction2[0].real+coefs[0,0].real, color='red')
plt.plot(time_points2[0], prediction2[0].real, color='red')
plt.show()

#%%
a, coefs, phis, prediction = fit_fmm_k(
    analytic_data_matrix=analytic_data_matrix2, n_back=n_back, max_iter=max_iter,
    time_points=time_points2, omega_grid=omega_grid,
    weights=np.ones(n_ch2), post_optimize=True, 
    omega_max = 0.999)

ch = 0
plt.plot(time_points2[0], analytic_data_matrix2[ch].real, color='blue')
plt.plot(time_points2[0], prediction[ch], color='purple')
plt.show()


#%% Funcion project_betas: ver que efectivamente se proyectan bien
from fit_fmm_k_restr import project_betas

n_back = 5
max_iter = 1
a, coefs, phis, prediction = fit_fmm_k(
    analytic_data_matrix=analytic_data_matrix2, n_back=n_back, max_iter=max_iter,
    time_points=time_points2, omega_grid=omega_grid,
    weights=np.ones(n_ch2), post_optimize=False)



#%%
# from fit_fmm_k_restr import inner_products_sum_restr
# from auxiliar_functions import split_complex

# def inner_products_sum_restr(splitted_a, analytic_data_matrix, time_points, k, a_parameters, weights, beta_min, beta_max):
    
#     a = splitted_a[1]*np.exp(1j*splitted_a[0])
#     a_parameters[k] = a
#     coefs = project_betas(analytic_data_matrix.real, time_points, a_parameters, beta_min, beta_max)
    
#     return -sum([weights[ch_i]*(np.abs(coefs_r) ** 2) for ch_i, coefs_r in enumerate(coefs)])

# val = inner_products_sum_restr(split_complex(a[-1]), analytic_data_matrix2, time_points2, 5, a, np.ones(10), np.pi-0.1, np.pi+0.1)

#%%
# weights = np.ones(10)
# np.sum([weights[ch_i]*(np.abs(coefs_r) ** 2) for ch_i, coefs_r in enumerate(coefs)])

#%%

from quadprog import solve_qp

# def generate_G(p, a, b):
#     G = np.zeros((2*p, 2*p+1))
#     h = np.zeros(2*p)
#     m1=m2=1
#     # Se puede hacer un bloque y repetirlo cuando a y b son fijos
#     if a > np.pi/2 and a < 3*np.pi/2:
#         m1 = -1
#     if b > np.pi/2 and b < 3*np.pi/2:
#         m2 = -1
#     for var in range(p):
#         G[2*var, 2*var+1] = m1*np.tan(a)
#         G[2*var, 2*var+2] = m1
#         G[2*var+1, 2*var+1] = -m2*np.tan(b)
#         G[2*var+1, 2*var+2] = -m2
#     return G, h

def generate_G(p, a, b):
    G = np.zeros((2*p, 2*p+1))
    h = np.zeros(2*p)

    for var in range(p):
        G[2*var, 2*var+1] = np.sin(a)
        G[2*var, 2*var+2] = -np.cos(a)
        G[2*var+1, 2*var+1] = -np.sin(b)
        G[2*var+1, 2*var+2] = np.cos(b)
    return G, h

n_back = len(a)-1
n_ch, n_obs = analytic_data_matrix2.real.shape
    
alphas = np.angle(a[1:]) + np.pi
omegas = (1-np.abs(a[1:]))/(1+np.abs(a[1:]))
    
AFD2FMM_matrix = transition_matrix(a)
phis = np.zeros((n_ch, n_back+1), dtype = 'complex')


#%%
from qpsolvers import solve_ls
ts = [2*np.arctan(omegas[i]*np.tan((time_points2[0] - alphas[i])/2)) for i in range(n_back)]
DM = np.column_stack([np.ones(n_obs)] + [np.column_stack([np.cos(ts[i]), np.sin(ts[i])]) for i in range(n_back)])

# P = 0.5 * DM.T @ DM  # Quadratic term
# q = (DM.T @ analytic_data_matrix2.real[0])

G, h = generate_G(n_back, np.pi-0.1, np.pi+0.1)

# from quadprog import solve_qp
# qp_problem = solve_qp(P, q, G.T, h)
RLS = solve_ls(DM, analytic_data_matrix2.real[0], G=G, h=h, solver='quadprog')

#%%


# Generates a matrix with p restrictions (between a and b)
def generate_G(p, a, b):
    G = np.zeros((2*p, 2*p+1))
    for var in range(p):
        G[2*var, 2*var+1] = np.sin(a)
        G[2*var, 2*var+2] = -np.cos(a)
        G[2*var+1, 2*var+1] = -np.sin(b)
        G[2*var+1, 2*var+2] = np.cos(b)
    return G

def project_betas(data_matrix, time_points, a, beta_min, beta_max):
    n_back = len(a) - 1
    n_ch, n_obs = data_matrix.shape
    
    # 1. AFD to Complex FMM
    G = generate_G(n_back, beta_min, beta_max)
    h = np.zeros(2 * n_back)
    
    # 2. Equivalent real FMM parameters
    alphas = np.angle(a[1:]) + np.pi
    omegas = (1 - np.abs(a[1:])) / (1 + np.abs(a[1:]))
    ts = [2*np.arctan(omegas[i]*np.tan((time_points[0] - alphas[i])/2)) for i in range(n_back)]
    DM = np.column_stack([np.ones(n_obs)] + [np.column_stack([np.cos(ts[i]), np.sin(ts[i])]) for i in range(n_back)])
    
    # 4. Allocate storage
    RLS = np.zeros((n_ch, 2 * n_back + 1))
    phis = np.zeros((n_ch, n_back + 1), dtype=np.complex128)
    
    # 5. Solve LSQ problem for all channels
    for ch_i in range(n_ch):
        RLS[ch_i] = solve_ls(DM, data_matrix[ch_i], G=G, h=h, solver='quadprog')
    
    # 6. Compute betas, amplitudes, and phis using vectorized operations
    betas = np.arctan2(-RLS[:, 2::2], RLS[:, 1::2])
    amplitudes = np.sqrt(RLS[:, 1::2] ** 2 + RLS[:, 2::2] ** 2)
    phis[:, 0] = RLS[:, 0]
    phis[:, 1:] = amplitudes * np.exp(1j * (betas - np.angle(a[1:]) + np.pi))

    # Return AFD coefs
    return np.dot(np.linalg.inv(transition_matrix(a)), phis.T).T

def project_betas_2(data_matrix, time_points, a, beta_min, beta_max):
    
    n_back = len(a)-1
    n_ch, n_obs = data_matrix.shape
    
    # 1. AFD to Complex FMM 
    AFD2FMM_matrix = transition_matrix(a)
    phis = np.zeros((n_ch, n_back+1), dtype = 'complex')
    G = generate_G(n_back, beta_min, beta_max)
    h = np.zeros(2*n_back)
    
    # 2. Equivalent real FMM parameters
    alphas = np.angle(a[1:]) + np.pi
    omegas = (1-np.abs(a[1:]))/(1+np.abs(a[1:]))
    
    # 3. To save restricted estimators
    betas = np.zeros((n_ch, n_back))
    amplitudes = np.zeros((n_ch, n_back))
    RLS = np.zeros((n_ch, 2*n_back + 1))
    
    # 4. Design matrix
    ts = [2*np.arctan(omegas[i]*np.tan((time_points2[0] - alphas[i])/2)) for i in range(n_back)]
    DM = np.column_stack([np.ones(n_obs)] + [np.column_stack([np.cos(ts[i]), np.sin(ts[i])]) for i in range(n_back)])
    
    for ch_i in range(n_ch):
        # RLS[ch_i] = solve_qp(DM, data_matrix[ch_i], G, h)[2]
        RLS[ch_i] = solve_ls(DM, data_matrix[ch_i], G=G, h=h, solver='quadprog')
        phis[ch_i, 0] = RLS[ch_i, 0]
        for k in range(n_back):
            betas[ch_i, k] = np.arctan2( -RLS[ch_i,2*k+2], RLS[ch_i,2*k+1])
            amplitudes[ch_i, k] = np.sqrt(RLS[ch_i,2*k+2]**2 +  RLS[ch_i,2*k+1]**2)
            phis[ch_i, k+1] = amplitudes[ch_i, k]*np.exp(1j*(betas[ch_i, k] - np.angle(a[k+1]) + np.pi ))
    coefs = np.dot(np.linalg.inv(AFD2FMM_matrix), phis.T).T

    return coefs

#%%
new_coefs = project_betas(analytic_data_matrix2.real, time_points2, a, np.pi-0.2, np.pi+0.2)
new_coefs_2 = project_betas_2(analytic_data_matrix2.real, time_points2, a, np.pi-0.2, np.pi+0.2)

#%%

prediction = predict(a, new_coefs_2, time_points2)
prediction2 = predictFMM(a, phis, time_points2)
plt.plot(time_points2[0], analytic_data_matrix2[0].real, color='blue')
plt.plot(time_points2[0], prediction[0].real, color='green')
# plt.plot(time_points2[0], prediction2[0].real, color='red')
plt.show()

#%%

from fit_fmm_k_restr import fit_fmm_k_restr_betas

n_back = 15
max_iter = 10
a, coefs, phis, prediction2 = fit_fmm_k_restr_betas(
    analytic_data_matrix=analytic_data_matrix2, n_back=n_back, max_iter=max_iter,
    time_points=time_points2, omega_grid=omega_grid,
    weights=np.ones(n_ch), post_optimize=False, beta_min=np.pi-0.1, beta_max=np.pi+0.1)

#%%

ch=0
plt.plot(time_points2[0], analytic_data_matrix2[ch].real, color='blue')
#plt.plot(time_points[0], prediction2[0].real+coefs[0,0].real, color='red')
plt.plot(time_points2[0], prediction2[ch].real, color='red')
plt.show()


