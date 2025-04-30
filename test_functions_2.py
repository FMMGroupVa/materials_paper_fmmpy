# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:08:15 2024

@author: Christian
"""

import numpy as np
import pandas as pd
import scipy.signal as sc
import matplotlib.pyplot as plt
import os
os.chdir(r'C:\Users\Christian\Documents\GitHub\PaquetePython')

import fit_fmm
# from  fit_fmm_unit import fit_fmm_unit
from auxiliar_functions import seq_times

#%%
df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\Patient1.csv')
df = df.iloc[:,350:850]

time_points = np.linspace(0, 2 * np.pi, num=df.shape[1]+1)[:-1]

analytic_data_matrix = sc.hilbert(df, axis = 1)
n_ch, n_obs = analytic_data_matrix.shape


omega_grid = np.linspace(0.01, 0.99, 50, False)

1/np.var(df, axis = 1)

np.var(analytic_data_matrix, axis = 1)

#%% COMPROBACION DEL AJUSTE 

from fit_fmm_k import fit_fmm_k
from auxiliar_functions import predict, predict2, predictFMM, seq_times, transition_matrix

time_points = seq_times(500)

n_back = 5
a, coefs, phis, prediction = fit_fmm_k(
    analytic_data_matrix=analytic_data_matrix, n_back=n_back, 
    time_points=time_points, omega_grid=omega_grid,
    weights=np.ones(n_ch), post_optimize=True)

# prediction2 = predict(a, coefs, time_points)

prediction2, coefs2 = predict2(a, analytic_data_matrix, time_points)


#%% PLOT DATA VS PREDICTION (UN CANAL)

plt.plot(time_points[0], analytic_data_matrix[0].real, color='blue')
#plt.plot(time_points[0], prediction2[0].real+coefs[0,0].real, color='red')
plt.plot(time_points[0], prediction2[0].real, color='red')
plt.show()

from auxiliar_functions import predict, predict2, predictFMM, seq_times, transition_matrix
from fit_fmm_k import fit_fmm_k, fit_fmm_k_mob 
from fit_fmm_k_restr import fit_fmm_k_restr, fit_fmm_k_mob_restr

#%% RESTRICCIONES

alpha_restrictions = [(np.pi, np.pi+2.5), (np.pi+2.5,2*np.pi), (np.pi+2.5,2*np.pi), (np.pi+2.5,2*np.pi), (0,np.pi)]
omega_restrictions = [(0.001,0.15), (0.001,0.1), (0.001,0.1), (0.001,0.1), (0.001,0.15)]

#%% RESTRICCIONES FIT

time_points = seq_times(500)

n_back = 5
max_iter = 10
a, coefs, phis, prediction = fit_fmm_k_restr(
    analytic_data_matrix=analytic_data_matrix, n_back=n_back, max_iter=max_iter,
    time_points=time_points, omega_grid=omega_grid,
    weights=np.ones(n_ch), post_optimize=False, 
    omega_min=0.001, omega_max = 0.999,
    alpha_restrictions=alpha_restrictions, omega_restrictions=omega_restrictions)

prediction2 = predict(a, coefs, time_points)
prediction3 = predict(np.delete(a, [1], axis=0), np.delete(coefs, [1], axis=1), time_points)
prediction4, coefs2 = predict2(np.delete(a, [1], axis=0), analytic_data_matrix, time_points)

# prediction2, coefs2 = predict2(a, analytic_data_matrix, time_points)

#%% PLOT DATA VS PREDICTION (UN CANAL)
ch=0
plt.plot(time_points[0], analytic_data_matrix[ch].real, color='blue')
plt.plot(time_points[0], prediction2[ch].real, color='red')
plt.plot(time_points[0], prediction3[ch].real, color='green')
plt.plot(time_points[0], prediction4[ch].real, color='purple')
plt.show()

#%%

from auxiliar_functions import szego, mobius

prediction = np.ones((n_ch, n_obs), dtype = complex)*coefs[:,0:1]
blaschke = np.ones((1, n_obs))

for k in range(1,6):
    comp = coefs[ch,k]*np.exp(1j*time_points)*szego(a[k], time_points)*blaschke
    blaschke = blaschke*mobius(a[k], time_points)
    plt.plot(time_points[0], comp[0].real, color='purple')
    plt.show()
    
    
#%%
from fit_fmm_k import fit_fmm_k, fit_fmm_k_mob, fit_fmm_k_mob_restr

alpha_grid = seq_times(50)
omega_grid = np.linspace(0.001, 0.5, 50, False)
n_back = 5
max_iter = 4
best_pars, best_pars_linear, remainder = fit_fmm_k_mob(data_matrix=analytic_data_matrix.real,  n_back=n_back, max_iter=max_iter, 
                                                   time_points=time_points, omega_grid=omega_grid, alpha_grid=alpha_grid, 
                                                   weights=np.ones(n_ch))

#%%
ch = 0
plt.plot(time_points[0], analytic_data_matrix[ch].real, color='blue')
plt.plot(time_points[0], analytic_data_matrix[ch].real - remainder[ch], color='purple')
plt.show()
    
#%% BETAS RESTRINGIDAS CON MOBIUS

alpha_grid = seq_times(50)
omega_grid = np.linspace(0.001, 0.5, 50, False)
n_back = 5
max_iter = 4
best_pars, best_pars_linear, remainder = fit_fmm_k_mob_restr(data_matrix=analytic_data_matrix.real,  n_back=n_back, max_iter=max_iter, 
                                                   time_points=time_points, omega_grid=omega_grid, alpha_grid=alpha_grid, 
                                                   weights=np.ones(n_ch),
                                                   beta_min=np.pi-1, beta_max=np.pi+1)

#%% 
ch = 0
plt.plot(time_points[0], analytic_data_matrix[ch].real, color='blue')
plt.plot(time_points[0], analytic_data_matrix[ch].real - remainder[ch], color='purple')
plt.show()
M, delta, gamma = best_pars_linear[0][:,5]

np.arctan2(-gamma, delta) % (2*np.pi)

#%%

df2 = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\Chroma_example.csv').T
df2 = df2.iloc[0:10,:]
time_points2 = seq_times(df2.shape[1])
analytic_data_matrix2 = sc.hilbert(df2, axis = 1)
n_ch2, n_obs2 = analytic_data_matrix2.shape

#%%
n_back = 30
max_iter = 1
a, coefs, phis, prediction = fit_fmm_k(
    analytic_data_matrix=analytic_data_matrix2, n_back=n_back, max_iter=max_iter,
    time_points=time_points2, omega_grid=omega_grid,
    weights=np.ones(n_ch2), post_optimize=False, 
    omega_max = 0.999)

prediction2 = predict(a, coefs, time_points2)
prediction3 = predict(np.delete(a, [1], axis=0), np.delete(coefs, [1], axis=1), time_points)
prediction4, coefs2 = predict2(np.delete(a, [1], axis=0), analytic_data_matrix, time_points)
ch=0
plt.plot(time_points2[0], analytic_data_matrix2[ch].real, color='blue')
plt.plot(time_points2[0], prediction2[ch].real, color='red')
plt.show()

#%%

alpha_grid = seq_times(50)
omega_grid = np.linspace(0.001, 0.5, 50, False)
n_back = 5
max_iter = 2
best_pars, best_pars_linear, remainder = fit_fmm_k_mob(data_matrix=analytic_data_matrix2.real,  n_back=n_back, max_iter=max_iter, 
                                                   time_points=time_points2, omega_grid=omega_grid, alpha_grid=alpha_grid, 
                                                   weights=np.ones(n_ch))

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

alpha_grid = seq_times(50)
omega_grid = np.linspace(0.001, 0.5, 50, False)
n_back = 15
max_iter = 10
best_pars, best_pars_linear, remainder = fit_fmm_k_mob_restr(data_matrix=analytic_data_matrix2.real,  n_back=n_back, max_iter=max_iter, 
                                                   time_points=time_points2, omega_grid=omega_grid, alpha_grid=alpha_grid, 
                                                   weights=np.ones(n_ch), beta_min=np.pi-0.5, beta_max=np.pi+0.5)

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

from fit_fmm_k import fit_fmm_k_restr_betas, project_betas

n_back = 3
max_iter = 1
a, coefs, phis, prediction2 = fit_fmm_k_restr_betas(
    analytic_data_matrix=analytic_data_matrix2, n_back=n_back, max_iter=max_iter,
    time_points=time_points2, omega_grid=omega_grid,
    weights=np.ones(n_ch), post_optimize=False, 
    beta_min=np.pi-0.5, beta_max=np.pi+0.5)

#%%
plt.plot(time_points2[0], analytic_data_matrix2[0].real, color='blue')
#plt.plot(time_points[0], prediction2[0].real+coefs[0,0].real, color='red')
plt.plot(time_points2[0], prediction2[0].real, color='red')
plt.show()


#%% Funcion project_betas: ver que efectivamente se proyectan bien
from fit_fmm_k import project_betas, predict

n_back = 5
max_iter = 1
a, coefs, phis, prediction = fit_fmm_k(
    analytic_data_matrix=analytic_data_matrix2, n_back=n_back, max_iter=max_iter,
    time_points=time_points2, omega_grid=omega_grid,
    weights=np.ones(n_ch), post_optimize=False)

#%%
new_coefs = project_betas(analytic_data_matrix2.real, time_points2, a, coefs, np.pi-0.01, np.pi+0.01)

#%%

prediction2 = predict(a, new_coefs, time_points2)

plt.plot(time_points2[0], analytic_data_matrix2[0].real, color='blue')
#plt.plot(time_points[0], prediction2[0].real+coefs[0,0].real, color='red')
plt.plot(time_points2[0], prediction2[0].real, color='red')
plt.plot(time_points2[0], prediction[0].real, color='green')
plt.show()

#%%

from fit_fmm_k import fit_fmm_k_restr_betas

n_back = 10
max_iter = 5
a, coefs, phis, prediction2 = fit_fmm_k_restr_betas(
    analytic_data_matrix=analytic_data_matrix2, n_back=n_back, max_iter=max_iter,
    time_points=time_points2, omega_grid=omega_grid,
    weights=np.ones(n_ch), post_optimize=False, beta_min=np.pi-0.01, beta_max=np.pi+0.01)

#%%
plt.plot(time_points2[0], analytic_data_matrix2[0].real, color='blue')
#plt.plot(time_points[0], prediction2[0].real+coefs[0,0].real, color='red')
plt.plot(time_points2[0], prediction[0].real, color='green')
plt.plot(time_points2[0], prediction2[0].real, color='red')
plt.show()

#%%

import numpy as np
import pandas as pd
import scipy.signal as sc
import matplotlib.pyplot as plt
import os
os.chdir(r'C:\Users\Christian\Documents\GitHub\PaquetePython')

import fit_fmm
from auxiliar_functions import seq_times

#%%
from fit_fmm_k import fit_fmm_k

df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\datosGOLEM.csv').iloc[:,1:].transpose()

time_points = seq_times(n_obs)
analytic_data_matrix = sc.hilbert(df, axis = 1)
n_ch, n_obs = analytic_data_matrix.shape

omega_grid = np.linspace(0.001, 0.2, 50, False)
n_back = 10

a, coefs, phis, prediction = fit_fmm_k(
    analytic_data_matrix=analytic_data_matrix, n_back=n_back, max_iter=1,
    time_points=time_points, omega_grid=omega_grid,
    weights=np.ones(n_ch), post_optimize=True, omega_min=0.01, omega_max=0.2)


#%%
for ch in range(n_ch):
    plt.plot(time_points[0], analytic_data_matrix[ch].real, color='blue')
    plt.plot(time_points[0], prediction[ch].real, color='red')
    plt.show()
    
#%%
from auxiliar_functions import szego, mobius
ch = 13
prediction = np.ones((n_ch, n_obs), dtype = complex)*coefs[:,0:1]
blaschke = np.ones((1, n_obs))
components = np.ones((10, n_obs))
for k in range(1,n_back+1):
    comp = coefs[ch,k]*np.exp(1j*time_points)*szego(a[k], time_points)*blaschke
    components[k-1] = comp.real
    blaschke = blaschke*mobius(a[k], time_points)
    plt.plot(time_points[0], comp[0].real, color='purple')
    plt.show()

np.savetxt("comps_ch13.csv", components.T, delimiter=",", fmt="%.4f")

#%%
df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\ejemploCristina.csv').transpose()

time_points = seq_times(n_obs)
analytic_data_matrix = sc.hilbert(df, axis = 1)
n_ch, n_obs = analytic_data_matrix.shape

omega_grid = np.linspace(0.05, 0.9, 25, False)
n_back = 4

a, coefs, phis, prediction = fit_fmm_k(
    analytic_data_matrix=analytic_data_matrix, n_back=n_back, max_iter=1,
    time_points=time_points, omega_grid=omega_grid,
    weights=np.ones(n_ch), post_optimize=True, omega_min=0.01, omega_max=0.99)

#%%
plt.plot(time_points[0], analytic_data_matrix[0].real, color='blue')
#plt.plot(time_points[0], prediction2[0].real+coefs[0,0].real, color='red')
plt.plot(time_points[0], prediction[0].real, color='green')
plt.show()

#%%





