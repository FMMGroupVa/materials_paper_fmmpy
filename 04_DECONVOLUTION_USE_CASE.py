# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.chdir(r'C:\Users\Christian\Documents\GitHub\PaquetePython')

from fit_fmm import fit_fmm


#%% Data: ECG beat 
# df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\example_chromatography.csv').T
df_raw = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\example_Fe2p.csv')

# Guardar la primera columna como eje x
x = df_raw.iloc[:, 0]  # Asumimos que esta columna contiene los valores de energía de enlace

# Procesar el resto del DataFrame
df = df_raw.iloc[:, 1:5].T / 1000 + 1
ltd = np.linspace(2.5, 0, num=df.shape[1])
df = df - ltd

# Plotear
plt.figure(figsize=(6, 4))
labels = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']
for i in range(df.shape[0]):
    plt.plot(x, df.iloc[i, :], label=labels[i])

plt.xlabel('Binding energy (eV)')
plt.ylabel('Intensity (u.a.)')
plt.legend()
# plt.gca().invert_xaxis()  # común en XPS
plt.tight_layout()
plt.savefig("Res/Fe2p.png", dpi=600, bbox_inches='tight')
plt.show()

#%% Fit FMM to data
res2 = fit_fmm(data_matrix=df, # Data
               n_back=8, max_iter=50, post_optimize=True,  # Fit options
               omega_min=0.01, omega_max=0.99) # Parameter control

#%% 
res2.plot_predictions(channels=[0], channel_names=[""],
                     dpi=600, width=2.8, height=2,
                     save_path="Res/Fe2p_example.png")
res2.plot_components(channels=[0], channel_names=[""],
                     dpi=600, width=2.8, height=2,
                     save_path="Res/Fe2p_comp_example.png")

#%% Fit FMM to data
from fit_fmm import fit_fmm

res3 = fit_fmm(data_matrix=df, # Data
              n_back=8, max_iter=50, post_optimize=True,  # Fit options
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
                      save_path="Res/Fe2p_example_restrB.png")
res3.plot_components(channels=[0], channel_names=[""],
                      dpi=600, width=2.8, height=2,
                      save_path="Res/Fe2p_comp_example_restrB.png")

# res3 = fit_fmm(data_matrix=df, # Data
#               n_back=8, max_iter=50, post_optimize=True,  # Fit options
#               omega_min=0.01, omega_max=0.1,
#               beta_min=np.pi-0.5, beta_max=np.pi+0.5)
# Da un error de P no positiva (QP optmization)
