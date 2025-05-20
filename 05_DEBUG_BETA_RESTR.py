# -*- coding: utf-8 -*-
"""
@author: Christian
"""

import numpy as np
import pandas as pd

import os
os.chdir(r'C:\Users\Christian\Documents\GitHub\PaquetePython')



#%% Data: ECG beat 
# df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\example_chromatography.csv').T
df_raw = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\example_Fe2p.csv')

# Guardar la primera columna como eje x
x = df_raw.iloc[:, 0]  # Asumimos que esta columna contiene los valores de energía de enlace

# Procesar el resto del DataFrame
df = df_raw.iloc[:, 1:5].T / 1000 + 1
ltd = np.linspace(2.5, 0, num=df.shape[1])
df = df - ltd

#%% Fit FMM to data
from fit_fmm import fit_fmm

n_back = 8
max_iter = 15

alpha_restr = np.array([(0,2.9) for i in range(n_back)])
omega_restr = np.array([(0.01,0.15) for i in range(n_back)])

res4 = fit_fmm(data_matrix=df, # Data
              n_back=n_back, max_iter=max_iter, post_optimize=True,  # Fit options
              omega_min=0.01, omega_max=0.12,
              beta_min=np.pi-0.3, beta_max=np.pi+0.3, 
              alpha_restrictions=alpha_restr, omega_restrictions=omega_restr)

#%%
res4.plot_predictions(channels=[1], channel_names=[""])
res4.plot_components(channels=[1])
print(res4.params['alpha'])
print(res4.params['omega'])
print(res4.R2)

# POR QUÉ VA MEJOR ESTE QUE EL QUE NO ESTA RESTRINGIDO????

#%%
res3 = fit_fmm(data_matrix=df, # Data
              n_back=n_back, max_iter=max_iter, post_optimize=True,  # Fit options
              omega_min=0.01, omega_max=0.1,
              beta_min=np.pi-0.3, beta_max=np.pi+0.3)

#%%
res3.plot_predictions(channels=[1], channel_names=[""])
res3.plot_components(channels=[1])
print(res3.params['alpha'])
print(res3.params['omega'])
print(res3.R2)

#%% Fit FMM to data
from fit_fmm import fit_fmm

n_back = 7
max_iter = 10

alpha_restr = np.array([(0,2.9) for i in range(n_back+1)])
omega_restr = np.array([(0.01,0.1) for i in range(n_back+1)])
groups = [1,1,2,3,4,5,6,7]
alpha_restr[0] = [0, 1.5]
alpha_restr[1] = [1.5, 2.9]

res4 = fit_fmm(data_matrix=df, # Data
              n_back=n_back, max_iter=max_iter, post_optimize=True,  # Fit options
              omega_min=0.01, omega_max=0.1,
              beta_min=np.pi-0.5, beta_max=np.pi+0.5, 
              alpha_restrictions=alpha_restr, omega_restrictions=omega_restr, 
              group_restrictions=groups)

#%%

res4.plot_predictions(channels=[1], channel_names=[""])
res4.plot_components(channels=[1])
print(res4.params['alpha'])
print(res4.params['omega'])
print(res4.R2)






