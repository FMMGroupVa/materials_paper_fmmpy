# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.chdir(r'C:\Users\Christian\Documents\GitHub\PaquetePython')

from fit_fmm import fit_fmm


#%% Data: ECG beat 
# df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\example_chromatography.csv').T
df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\example_Fe2p.csv')
df = df.iloc[:,1:5].T/1000+1
ltd = np.linspace(2.5, 0, num=df.shape[1])
df = df-ltd

plt.plot(df.iloc[0,:])
plt.plot(df.iloc[1,:])
plt.plot(df.iloc[2,:])
plt.plot(df.iloc[3,:])
#%% Fit FMM to data
res2 = fit_fmm(data_matrix=df, # Data
              n_back=8, max_iter=50, post_optimize=True,  # Fit options
              omega_min=0.01, omega_max=0.99) # Parameter control

res2.plot_predictions(channels=[0], channel_names=[""],
                     dpi=600, width=2.8, height=2,
                     save_path="Res/Fe2p_example.png")
res2.plot_components(channels=[0], channel_names=[""],
                     dpi=600, width=2.8, height=2,
                     save_path="Res/Fe2p_comp_example.png")

#%% Fit FMM to data
from fit_fmm import fit_fmm

res3 = fit_fmm(data_matrix=df, # Data
              n_back=7, max_iter=50, post_optimize=True,  # Fit options
              omega_min=0.01, omega_max=0.1,
              beta_min=np.pi-0.25, beta_max=np.pi+0.25)

#%%

res3.plot_predictions(channels=[1], channel_names=[""])
res3.plot_components(channels=[1])
print(res3.params['beta'])
print(res3.R2)

#%%

res3.plot_predictions(channels=[0], channel_names=[""],
                      dpi=600, width=2.8, height=2,
                      save_path="Res/Fe2p_example_restr.png")
res3.plot_components(channels=[0], channel_names=[""],
                      dpi=600, width=2.8, height=2,
                      save_path="Res/Fe2p_comp_example_restr.png")

# res3 = fit_fmm(data_matrix=df, # Data
#               n_back=8, max_iter=50, post_optimize=True,  # Fit options
#               omega_min=0.01, omega_max=0.1,
#               beta_min=np.pi-0.5, beta_max=np.pi+0.5)
# Da un error de P no positiva (QP optmization)
