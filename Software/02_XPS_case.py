# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyFMM.fit_fmm import fit_fmm

df_raw = pd.read_csv('Data/XPS_Fe2p_data.csv')
x = df_raw.iloc[:, 0]  
df = df_raw.iloc[:, 1:5].T

plt.figure(figsize=(6, 4))
plt.plot(x, df.iloc[0, :])
plt.xlabel('Binding energy (eV)')
plt.ylabel('Intensity (u.a.)')
plt.tight_layout()
plt.grid()
plt.savefig("Results/Figures/Fe2p.png", dpi=600, bbox_inches='tight')
plt.show()

spectrum = df.iloc[0, :]

n_back=7
max_iter=15

res2 = fit_fmm(data_matrix=spectrum, # Data
               n_back=n_back, max_iter=max_iter, post_optimize=True,  # Fit options
               omega_min=0.01) # Parameter control

res2.plot_predictions(channels=[0], channel_names=[""],
                     dpi=300, width=2.8, height=2,
                     save_path="Results/Figures/Fe2p_example.png")
res2.plot_components(channels=[0], channel_names=[""],
                     dpi=300, width=2.8, height=2,
                     save_path="Results/Figures/Fe2p_comp_example.png")



alpha_restr = np.array([(0.1,2.9) for i in range(n_back)])

res3 = fit_fmm(data_matrix=spectrum, # Data
              n_back=n_back, max_iter=max_iter, post_optimize=True,  # Fit options
              omega_min=0.01, omega_max=0.1,
              length_alpha_grid=100, alpha_restrictions=alpha_restr,
              beta_min=np.pi-0.25, beta_max=np.pi+0.25)

res3.plot_predictions(channels=[0], channel_names=[""],
                     dpi=300, width=2.8, height=2,
                     save_path="Results/Figures/Fe2p_example_restr.png")
res3.plot_components(channels=[0], channel_names=[""],
                     dpi=300, width=2.8, height=2,
                     save_path="Results/Figures/Fe2p_comp_example_restr.png")







