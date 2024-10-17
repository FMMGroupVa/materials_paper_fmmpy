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

afd_abs = np.linspace(0, 1, 50, False)
omega_grid = (1-afd_abs)/(1+afd_abs)*np.exp(0*1j)
omega_grid = np.reshape(omega_grid, (50))

#%% COMPROBACION DEL AJUSTE 

from fit_fmm_k import fit_fmm_k
from auxiliar_functions import predict, predictFMM, seq_times, transition_matrix

time_points = seq_times(500)

n_back = 5
a, coefs, prediction = fit_fmm_k(
    analytic_data_matrix=analytic_data_matrix, n_back=n_back, 
    time_points=time_points, omega_grid=omega_grid,
    weights=np.ones(n_ch), post_optimize=True)

prediction2 = predict(a, coefs, time_points)


#%% PLOT DATA VS PREDICTION (UN CANAL)

plt.plot(time_points[0], analytic_data_matrix[0].real, color='blue')
#plt.plot(time_points[0], prediction2[0].real+coefs[0,0].real, color='red')
plt.plot(time_points[0], prediction2[0].real, color='red')
plt.show()

#%%
from auxiliar_functions import predict, predictFMM, transition_matrix

AFD_to_FMM_matrix = transition_matrix(a[1:])
phis = np.dot(AFD_to_FMM_matrix, coefs.T).T

#%%
yFMM = predictFMM(a, phis, time_points)

plt.plot(time_points[0], analytic_data_matrix[0].real, color='blue')
plt.plot(time_points[0], analytic_data_matrix[0].real, color='blue')
plt.plot(time_points[0], prediction2[0].real, color='red')
plt.show()





