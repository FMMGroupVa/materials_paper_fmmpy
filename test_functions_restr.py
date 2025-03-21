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
n_ch, n_obs = analytic_data_matrix2.shape

omega_grid = np.linspace(0.001, 0.1, 50, False)

#%% Unrestricted Optimization  - AFD
n_back = 30
max_iter = 1
a, coefs, phis, prediction = fit_fmm_k(
    analytic_data_matrix=analytic_data_matrix2, n_back=n_back, max_iter=max_iter,
    time_points=time_points2, omega_grid=omega_grid,
    weights=np.ones(n_ch), post_optimize=False, 
    omega_max = 0.999)

ch = 0
plt.plot(time_points2[0], analytic_data_matrix2[ch].real, color='blue')
plt.plot(time_points2[0], prediction[ch], color='purple')
plt.show()

#%% Unrestricted Optimization - FMM

alpha_grid = seq_times(50)
omega_grid = np.linspace(0.001, 0.5, 50, False)
n_back = 5
max_iter = 2
best_pars, best_pars_linear, remainder = fit_fmm_k_mob(data_matrix=analytic_data_matrix2.real,  n_back=n_back, max_iter=max_iter, 
                                                   time_points=time_points2, omega_grid=omega_grid, alpha_grid=alpha_grid, 
                                                   weights=np.ones(n_ch))


#%%

for k in range(n_back):
    alpha, omega = best_pars[k]
    ts = 2*np.arctan(omega*np.tan((time_points2[0] - alpha)/2)) 
    delta, gamma = best_pars_linear[k][1:3,ch]
    comp = delta*np.cos(ts) + gamma*np.sin(ts)
    plt.plot(time_points2[0], comp-comp[0], color='purple')
    
plt.show()

#%% Restricted Optimization - FMM

alpha_grid = seq_times(50)
omega_grid = np.linspace(0.001, 0.5, 50, False)
n_back = 10
max_iter = 1
inicio = time.time()
best_pars, best_pars_linear, remainder = fit_fmm_k_mob_restr(data_matrix=analytic_data_matrix2.real,  n_back=n_back, max_iter=max_iter, 
                                                   time_points=time_points2, omega_grid=omega_grid, alpha_grid=alpha_grid, 
                                                   weights=np.ones(n_ch), beta_min=np.pi-0.5, beta_max=np.pi+0.5)
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



#%% Restricted Optimization - AFD

from fit_fmm_k_restr import fit_fmm_k_restr_betas
import time

n_back = 1
max_iter = 1

start = time.perf_counter()
omega_grid = np.linspace(0.01, 0.15, 50, False)
a, coefs, phis, prediction2 = fit_fmm_k_restr_betas(
    analytic_data_matrix=analytic_data_matrix2, n_back=n_back, max_iter=max_iter,
    time_points=time_points2, omega_grid=omega_grid,
    weights=np.ones(n_ch), post_optimize=False, omega_min=0.005, omega_max=0.2,
    beta_min=np.pi-0.1, beta_max=np.pi+0.1)
end = time.perf_counter()

print(f"Tiempo de ejecución: {end - start:.6f} segundos")

#%%

ch=0
plt.plot(time_points2[0], analytic_data_matrix2[ch].real, color='blue')
plt.plot(time_points2[0], prediction2[ch].real, color='red')
plt.show()

#%%

from fit_fmm_k_restr import fit_fmm_k_restr_betas
import cProfile

n_back = 5
max_iter = 1

cProfile.run('fit_fmm_k_restr_betas(analytic_data_matrix=analytic_data_matrix2, n_back=n_back, max_iter=max_iter,time_points=time_points2, omega_grid=omega_grid,weights=np.ones(n_ch), post_optimize=False, beta_min=np.pi-0.1, beta_max=np.pi+0.1)')

