# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 12:10:21 2025

@author: Christian
"""

import numpy as np
import pandas as pd

import os
os.chdir(r'C:\Users\Christian\Documents\GitHub\PaquetePython')

from fit_fmm import fit_fmm

#%% Data: ECG beat 
df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\Patient1.csv', header=None)
df = df.iloc[:,350:850]

#%% Fit FMM to data
P_alpha = (4.2, 5.4)
P_ome = (0.05,0.25)
QRS_alpha = (5.4, 6.2)
QRS_ome = (0.01, 0.15)
T_alpha = (0, 3.14)
T_ome = (0.1, 0.5)
# Param restriction arguments:
alpha_restr = np.array([P_alpha, QRS_alpha, QRS_alpha, QRS_alpha, T_alpha])
omega_restr = np.array([P_ome, QRS_ome, QRS_ome, QRS_ome, T_ome])
# alpha_restr = np.array([P_alpha, QRS_alpha, QRS_alpha, T_alpha])
# omega_restr = np.array([P_ome, QRS_ome, QRS_ome, T_ome])

res = fit_fmm(data_matrix=df, # Data
              n_back=5, max_iter=8, post_optimize=True,  # Fit options
              alpha_restrictions=alpha_restr, omega_restrictions=omega_restr,
              omega_min=0.01, omega_max=0.5) # Parameter control

#%% Print model summary:
print(res)
    

#%% Plot results:
lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
res.plot_components(channel_names=lead_names, n_cols=3,
                     width=5.9, height=5, dpi=300, save_path="Res/components.png")

#%% Plot results:
lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
res.plot_predictions(channel_names=lead_names, n_cols=3,
                     width=5.9, height=5, dpi=300, save_path="Res/3leadsexample.png")

#%% Plot results:
res.plot_residuals(channel_names=lead_names, n_cols=4,
                   width=5.9, height=5, dpi=600, save_path="Res/ECGresiduals.png")

#%% Plot results:
alpha_ci, omega_ci, delta_ci, gamma_ci = res.conf_intervals(0.95, method=2)
print(alpha_ci)

res.show_conf_intervals()




