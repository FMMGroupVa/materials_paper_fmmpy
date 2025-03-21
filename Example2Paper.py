# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 09:26:00 2025

@author: Christian
"""
import numpy as np
import pandas as pd
import scipy.signal as sc
import matplotlib.pyplot as plt
import os

from fit_fmm_k import fit_fmm_k
from auxiliar_functions import predict, seq_times

os.chdir(r'C:\Users\Christian\Documents\GitHub\PaquetePython')

import fit_fmm
# from  fit_fmm_unit import fit_fmm_unit
from auxiliar_functions import seq_times

# Define the complex numbers for 'as' and 'cs'
a1 = np.array([0+0j, 0.6800 + 0.5200j, 0.3900 + 0.8100j, -0.1300 - 0.8700j, 0.5500 + 0.1000j])
c1 = np.array([[0+0j, 0.1440 + 0.5197j, -1.6387 - 0.0142j, -0.7601 - 1.1555j, -0.8188 - 0.0095j]])

a2 = np.array([0+0j,-0.4900 - 0.8000j, 0.3100 + 0.1400j, -0.9400 - 0.2900j, 0.2300 - 0.6900j])
c2 = np.array([[0+0j,1.0470 + 0.55587j, -0.2269 - 1.1203j, -0.1625 - 1.5327j, 0.6901 - 1.0979j]])

a3 = np.array([0+0j,-0.1800 + 0.7700j, -0.0200 - 0.1800j, 0.1000 + 0.2400j, 0.1800 - 0.5300j])
c3 = np.array([[0+0j, 0.1097 + 0.4754j, 1.1287 + 1.1741j, -0.2900 + 0.1269j, 1.2616 - 0.6568j]])

a4 = np.array([0.4100 - 0.0250j, -0.6200 - 0.2700j, 0.4400 + 0.5700j])
c4 = np.array([[-0.2060 + 0.0821j, -0.1420 - 0.6210j, 0.0521 + 0.7287j]])

time_points = seq_times(1024)

#%%

model1 = predict(a1, c1, time_points)
model2 = predict(a2, c2, time_points)
model3 = predict(a3, c3, time_points)
model4 = c4[0,0]/(1-np.conj(a4[0])*np.exp(1j*time_points)) + c4[0,1]/(1-np.conj(a4[1])*np.exp(1j*time_points)) + c4[0,2]/(1-np.conj(a4[2])*np.exp(1j*time_points)) 
plt.plot(time_points[0], model1[0].real, color='blue')
plt.show()
plt.plot(time_points[0], model2[0].real, color='blue')
plt.show()
plt.plot(time_points[0], model3[0].real, color='blue')
plt.show()
plt.plot(time_points[0], model4[0].real, color='blue')
plt.show()

#%% Example 1

# Para el paper: subir el grid de omegas a 10.000 y max_iter > 500
omega_grid = np.linspace(0.001, 0.5, 2000)
aAFD1, cAFD1, _, prediction = fit_fmm_k(
    analytic_data_matrix=model1[0], n_back=4, max_iter=250, time_points=time_points, 
    omega_grid=omega_grid, weights=np.ones(1), post_optimize=False)

aAFD1
#%% 
plt.plot(time_points[0], model1[0].real, color='blue')
plt.plot(time_points[0], prediction[0].real, color='red')

#%% Example 2

# Para el paper: subir el grid de omegas a 10.000 y max_iter > 500
omega_grid = np.linspace(0.001, 0.999, 3999)
aAFD2, cAFD2, _, prediction = fit_fmm_k(
    analytic_data_matrix=model2[0], n_back=4, max_iter=250, time_points=time_points, 
    omega_grid=omega_grid, weights=np.ones(1), post_optimize=False)

aAFD2

# -0.93952734-0.29130967j, -0.93952734-0.29130967j,  0.30933517+0.13485398j, 0.22817482-0.69069505j
#%%
prediction2 = predict(aAFD2, cAFD2, time_points)
plt.plot(time_points[0], model2[0].real, color='blue')
plt.plot(time_points[0], prediction2[0].real, color='red')

#%% Example 3

# Para el paper: subir el grid de omegas a 10.000 y max_iter > 500
omega_grid = np.linspace(0.05, 0.95, 500)
aAFD3, cAFD3, _, prediction = fit_fmm_k(
    analytic_data_matrix=model3[0], n_back=4, max_iter=50, time_points=time_points, 
    omega_grid=omega_grid, weights=np.ones(1), post_optimize=False)

aAFD3

# -0.03228541-0.17340461j, 0.19316053-0.53984752j,  0.10894471+0.23785305j, 0.14215852-0.63306062j
# -0.12050102-0.23939441j, -0.17301702+0.69072264j,  0.22627898+0.29364699j, 0.22620445-0.55572615j
#%%
plt.plot(time_points[0], model3[0].real, color='blue')
plt.plot(time_points[0], prediction[0].real, color='red')
        
#%%

# Para el paper: subir el grid de omegas a 10.000 y max_iter > 500
omega_grid = np.linspace(0.001, 0.999, 500)
aAFD4, cAFD4, _, prediction = fit_fmm_k(
    analytic_data_matrix=model4[0], n_back=3, max_iter=250, time_points=time_points, 
    omega_grid=omega_grid, weights=np.ones(1), post_optimize=False)

aAFD4

# 0.41541988-0.03064321j, -0.61981194-0.27020564j,  0.4397316 +0.57064892j
#%%
plt.plot(time_points[0], model4[0].real, color='blue')
plt.plot(time_points[0], prediction[0].real, color='red')


















