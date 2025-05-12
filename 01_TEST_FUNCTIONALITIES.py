# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 11:59:38 2025

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
res = fit_fmm(data_matrix=df, # Data
              n_back=5, max_iter=10, post_optimize=True,  # Fit options
              omega_min=0.01, omega_max=0.99) # Parameter control

res.plot_predictions() # By default: dpi = 100 (low)

#%% Plot results:

lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
res.plot_predictions(channels = [0,1,2], channel_names=lead_names[0:3], n_cols=3,
                     width=5.9, height=2, dpi=600, save_path="Res/3leadsexample.png")

#%%
res.plot_predictions(channels=[1], channel_names=lead_names[1],
                     dpi=600, save_path="Res/leadIIexample.png")

#%% How to define restrictions on alphas/omegas

from fit_fmm import fit_fmm

alpha_restr = (np.array([(0,2.5), (2.5,3.5), (2.5,3.5), (2.5,3.5), (3.5,6)]) + np.pi) % (2*np.pi)
omega_restr = np.array([(0.05,0.15), (0.005,0.1), (0.005,0.1), (0.005,0.1), (0.05,0.2)])

res = fit_fmm(data_matrix=df, 
              n_back=5, max_iter=5, post_optimize=True, omega_min=0.01, omega_max=0.99,
              alpha_restrictions=alpha_restr, omega_restrictions=omega_restr) 
res.params['alpha']

#%% Plot results:
res.plot_predictions() # By default: dpi = 100 (low)
res.plot_predictions(channels = [0,1,2,3], dpi=300)
res.params['omega']

#%% Plot results:
res.plot_predictions() # By default: dpi = 100 (low)
res.plot_predictions(channels = [1], dpi=300)

#%% Try block restrictions
############################################# DEBUG #############################################
    
from fit_fmm import fit_fmm
alpha_restr = (np.array([(0,1.5), (1.5,2.5), (2.5,3.5), (2.5,3.5), (2.5,3.5), (3.5,6)]) + np.pi) % (2*np.pi)
omega_restr = np.array([(0.05,0.15),(0.025,0.15), (0.005,0.1), (0.005,0.1), (0.005,0.1), (0.05,0.2)])
gr_restr = [1,1,2,3,4,5]

res = fit_fmm(data_matrix=df, 
              n_back=5, max_iter=5, post_optimize=False, omega_min=0.01, omega_max=0.99,
              alpha_restrictions=alpha_restr, omega_restrictions=omega_restr, group_restrictions=gr_restr) 

res.plot_predictions() # By default: dpi = 100 (low)
res.plot_predictions(channels = [0,1,2,3], dpi=300)
print(res.params['alpha'])
print(res.params['omega'])










