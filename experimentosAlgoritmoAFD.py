# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:39:29 2025

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

#%% EJEMPLO 1
# df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\Patient1.csv', header=None)
# df = df.iloc[:,350:850]

df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\ICARE_0357.csv')
df = df.transpose()

time_points = np.linspace(0, 2 * np.pi, num=df.shape[1]+1)[:-1]

analytic_data_matrix = sc.hilbert(df, axis = 1)
n_ch, n_obs = analytic_data_matrix.shape


omega_grid = np.exp(np.linspace(np.log(0.001), np.log(0.25), 50))
omega_grid = np.reshape(omega_grid, (50))

#%% COMPROBACION DEL AJUSTE 

from fit_fmm_k import fit_fmm_k
from auxiliar_functions import predict, seq_times

time_points = seq_times(n_obs)

n_back = 15
a_model, coefs_model, phis, prediction = fit_fmm_k(
    analytic_data_matrix=analytic_data_matrix, n_back=n_back, max_iter=1,
    time_points=time_points, omega_grid=omega_grid,
    weights=np.ones(n_ch), post_optimize=False)

prediction2 = predict(a_model, coefs_model, time_points)

#%%
plt.plot(np.angle(a_model), np.abs(a_model), 'o', color='red')
plt.show()

channel = 0
plt.plot(time_points[0], analytic_data_matrix[channel].real, color='blue')
plt.plot(time_points[0], prediction2[channel].real, color='red')

for xc in (np.angle(a_model[1:])%(2*np.pi)):
    plt.axvline(x=xc)
    
model1 = prediction2[channel]

#%%

# pd.DataFrame(model1.real).to_csv('EEG_sin_ruido.csv', index=False)

#%% COMPROBACION DEL AJUSTE 
max_comps = 15

R2j = np.zeros(max_comps)
for j in range(1,max_comps+1):
    a, coefs, phis, prediction = fit_fmm_k(
        analytic_data_matrix=model1, n_back=j, max_iter=1,
        time_points=time_points, omega_grid=omega_grid,
        weights=np.ones(n_ch), post_optimize=False)
    
    plt.plot(time_points[0], model1.real, color='blue')
    plt.plot(time_points[0], prediction[0].real, color='red')
    plt.show()
    
    R2j[j-1] = 1-np.var(model1.real-prediction.real[0])/np.var(model1.real)
    
# pd.DataFrame(prediction.real[0]).to_csv('EEG_AFD_ajuste.csv', index=False)




#%%

# Para el paper: subir el grid de omegas a 10.000 y max_iter > 500
omega_grid = np.exp(np.linspace(np.log(0.001), np.log(0.15), 50))
omega_grid = np.reshape(omega_grid, (50))

a2, coefs2, _, prediction = fit_fmm_k(
    analytic_data_matrix=model1, n_back=15, max_iter=2000, time_points=time_points, 
    omega_grid=omega_grid, weights=np.ones(1), post_optimize=True, omega_max=0.2)

plt.plot(time_points[0], model1.real, color='blue')
plt.plot(time_points[0], prediction[0].real, color='red')


#%%

pp = predict(a_model, coefs_model, time_points)
pp2 = predict(a2, coefs2, time_points)
plt.plot(time_points[0], model1.real, '-',color='black')
plt.plot(time_points[0], pp[0].real, '--',color='red')
plt.plot(time_points[0], pp2[0].real, '--',color='blue')
plt.show()

#%%

plt.plot(np.angle(a2[1:]), abs(a2[1:]), 'o')
plt.plot(np.angle(a_model[1:]), abs(a_model[1:]), 'o')
plt.show()

from auxiliar_functions import transition_matrix, predictFMM, mobius

AFD2FMM_matrix = transition_matrix(a2)
  
phis = np.dot(AFD2FMM_matrix, coefs2.T).T

predfmm = predictFMM(a2, phis, time_points)

plt.plot(time_points[0], model1.real, '-',color='black')
plt.plot(time_points[0], predfmm[0].real, '--',color='red')
plt.show()



#%% EJEMPLO 2

# Define the complex numbers for 'as' and 'cs'
# a_s = np.array([0+0j, 0.6800 + 0.5200j, 0.3900 + 0.8100j, -0.1300 + 0.8700j, 0.5500 + 0.1000j])
# c_s = np.array([[0+0j, 0.1440 + 0.5197j, -1.6387 - 0.0142j, -0.7601 - 1.1555j, -0.8188 - 0.0095j]])

# # Generate times and compute z
# time_points = seq_times(1024)

# pp = predict(a_s, c_s, time_points)
# plt.plot(time_points[0], pp[0].real, color='blue')

# #%%

# # Para el paper: subir el grid de omegas a 10.000 y max_iter > 500
# omega_grid = np.linspace(0.001, 0.2, 500)
# a2, coefs2, _, prediction = fit_fmm_k(
#     analytic_data_matrix=pp[0], n_back=15, max_iter=250, time_points=time_points, 
#     omega_grid=omega_grid, weights=np.ones(1), post_optimize=False)

# plt.plot(time_points[0], model1.real, color='blue')
# plt.plot(time_points[0], prediction[0].real, color='red')
