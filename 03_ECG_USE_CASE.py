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
res = fit_fmm(data_matrix=df, # Data
              n_back=5, max_iter=10, post_optimize=True,  # Fit options
              omega_min=0.01, omega_max=0.99) # Parameter control

#%% Plot results:
lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
res.plot_predictions(channels = [0,1,2], channel_names=lead_names[0:3], n_cols=3,
                     width=5.9, height=2, dpi=600, save_path="Res/3leadsexample.png")





