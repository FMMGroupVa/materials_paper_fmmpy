# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:54:36 2025

@author: Christian
"""

import numpy as np
import pandas as pd

import os
os.chdir(r'C:\Users\Christian\Documents\GitHub\PaquetePython')

from fit_fmm import fit_fmm

#%% Data: ECG beat 
df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\Patient1.csv', header=None)
df = df.iloc[1,340:840]

#%% Fit FMM to data
res = fit_fmm(data_matrix=df, # Data
              n_back=5, max_iter=20, post_optimize=True,  # Fit options
              omega_min=0.01, omega_max=0.99) # Parameter control
print(res)

res.plot_predictions(dpi=300)
res.plot_components(dpi=300)

res.conf_intervals()
#%%

print(res.params['beta'])

#%% Fit FMM to data
from fit_fmm import fit_fmm

P_alpha = (4.2, 5.4)
P_ome = (0.05,0.25)
QRS_alpha = (5.4, 6.2)
QRS_ome = (0.01, 0.15)
T_alpha = (0, 3.14)
T_ome = (0.1, 0.5)
# Param restriction arguments:
alpha_restr = np.array([P_alpha, QRS_alpha, QRS_alpha, QRS_alpha, T_alpha])
omega_restr = np.array([P_ome, QRS_ome, QRS_ome, QRS_ome, T_ome])

res = fit_fmm(data_matrix=df, # Data
              n_back=5, max_iter=20, post_optimize=True,  # Fit options
              alpha_restrictions=alpha_restr, omega_restrictions=omega_restr,
              omega_min=0.01, omega_max=0.5) # Parameter control

print(res.params['alpha'])
print(res.params['beta'])
