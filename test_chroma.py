# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:56:25 2024

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
df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\example_chromatography.csv').T

#%%
time_points = seq_times(df.shape[1])

analytic_data_matrix = sc.hilbert(df, axis = 1)
n_ch, n_obs = analytic_data_matrix.shape
afd_abs = np.linspace(0, 1, 50, False)
omega_grid = (1-afd_abs)/(1+afd_abs)*np.exp(0*1j)
omega_grid = np.reshape(omega_grid, (50))

#%% COMPROBACION DEL AJUSTE 

from fit_fmm_k import fit_fmm_k
from auxiliar_functions import predict

n_back = 15
a, coefs, phis, prediction = fit_fmm_k(
    analytic_data_matrix=analytic_data_matrix, n_back=n_back, 
    time_points=time_points, omega_grid=omega_grid,
    weights=np.ones(n_ch), post_optimize=True)

prediction2 = predict(a, coefs, time_points)


#%% PLOT DATA VS PREDICTION (UN CANAL)

nch = 20
plt.plot(time_points[0], analytic_data_matrix[nch].real, color='blue')
#plt.plot(time_points[0], prediction2[0].real+coefs[0,0].real, color='red')
plt.plot(time_points[0], prediction2[nch].real, color='red')
plt.show()


















