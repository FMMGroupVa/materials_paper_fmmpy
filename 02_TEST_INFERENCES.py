# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 18:20:43 2025

@author: Christian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir(r'C:\Users\Christian\Documents\GitHub\PaquetePython')

from fit_fmm import fit_fmm

#%% Data: ECG beat 
# df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\Patient1.csv', header=None)
# df = df.iloc[:,350:850]

df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\exampleData.csv').T

#%% Fit FMM to data
res = fit_fmm(data_matrix=df, # Data
              n_back=5, max_iter=30, post_optimize=True,  # Fit options
              omega_min=0.01, omega_max=0.4) # Parameter control

print(res)
res.plot_residuals()
res.plot_predictions()
#%%

alpha_ci, omega_ci, delta_ci, gamma_ci = res.conf_intervals(0.95, method=1)
print("Method 1")
print(omega_ci[0])
print(omega_ci[1])
print((omega_ci[1]-omega_ci[0])/4)
print(alpha_ci[0])
print(alpha_ci[1])
print((alpha_ci[1]-alpha_ci[0])/4)

alpha_ci, omega_ci, delta_ci, gamma_ci = res.conf_intervals(0.95, method=2)
print("Method 2")
print(omega_ci[0])
print(omega_ci[1])
print((omega_ci[1]-omega_ci[0])/4)
print(alpha_ci[0])
print(alpha_ci[1])
print((alpha_ci[1]-alpha_ci[0])/4)

alpha_ci, omega_ci, delta_ci, gamma_ci = res.conf_intervals(0.95, method=3)
print("Method 3")
print(omega_ci[0])
print(omega_ci[1])
print((omega_ci[1]-omega_ci[0])/4)
print(alpha_ci[0])
print(alpha_ci[1])
print((alpha_ci[1]-alpha_ci[0])/4)

#%%

rr = res.data-res.prediction

plt.plot(res.time_points[0], rr[0])

np.std(rr,axis=1)

np.sqrt(np.var(rr,axis=1))
np.var(res.data,axis=1)

