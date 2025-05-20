# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 18:20:43 2025

@author: Christian
"""

import numpy as np
import pandas as pd

import os
os.chdir(r'C:\Users\Christian\Documents\GitHub\PaquetePython')

from fit_fmm import fit_fmm


#%% Data: ECG beat 
# df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\Patient1.csv', header=None)
# df = df.iloc[:,350:850]

df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\exampleData.csv').T

#%% Fit FMM to data
res = fit_fmm(data_matrix=df, # Data
              n_back=5, max_iter=20, post_optimize=True,  # Fit options
              omega_min=0.01, omega_max=0.99) # Parameter control

print(res)
#%%

alpha_ci, omega_ci, delta_ci, gamma_ci = res.conf_intervals(0.95)
print(omega_ci)
















