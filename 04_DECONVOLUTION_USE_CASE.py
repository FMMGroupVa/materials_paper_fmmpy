# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 16:18:18 2025

@author: Christian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.chdir(r'C:\Users\Christian\Documents\GitHub\PaquetePython')

from fit_fmm import fit_fmm


#%% Data: ECG beat 
# df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\example_chromatography.csv').T
df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\example_Fe2p.csv')
df = df.iloc[:,1:5].T

#%% Fit FMM to data
res = fit_fmm(data_matrix=df, # Data
              n_back=6, max_iter=20, post_optimize=False,  # Fit options
              omega_min=0.01, omega_max=0.4) # Parameter control


res.plot_predictions(channels=[0])
res.plot_components(channels=[0])

#%%

waves = res.get_waves(200)
plt.plot(np.sum(waves[0], axis=0))

#%% Fit FMM to data
res = fit_fmm(data_matrix=df, # Data
              n_back=1, max_iter=1, post_optimize=False,  # Fit options
              omega_min=0.01, omega_max=0.1,
              beta_min=np.pi-0.1, beta_max=np.pi+0.1)

#%%

res.plot_predictions(channels=[0])
res.plot_components(channels=[0])
res.params['beta']
