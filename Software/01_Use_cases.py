# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyFMM.fit_fmm import fit_fmm


############################## CASE 1 ##############################

# Data: ECG beat 
df = pd.read_csv('Data/ECG_data.csv', header=None)
df = df.iloc[:,400:800]

# Fit FMM to data
P_alpha = (4.2, 5.4)
P_ome = (0.05,0.25)
QRS_alpha = (5.4, 6.2)
QRS_ome = (0.01, 0.10)
T_alpha = (0, 3.14)
T_ome = (0.1, 0.5)

# Param restriction arguments:
alpha_restr = np.array([P_alpha, QRS_alpha, QRS_alpha, QRS_alpha, T_alpha])
omega_restr = np.array([P_ome, QRS_ome, QRS_ome, QRS_ome, T_ome])

res = fit_fmm(data_matrix=df, # Data
              n_back=5, max_iter=8, post_optimize=True,  # Fit options
              alpha_restrictions=alpha_restr, omega_restrictions=omega_restr,
              omega_min=0.01, omega_max=0.5) # Parameter control

# Print model summary:
print(res)
    
# Plot results fit:
lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
res.plot_predictions(channel_names=lead_names, n_cols=3,
                     width=5.9, height=5, dpi=300, save_path="Results/Figures/ECGfit.png")

# Plot results residuals:
res.plot_residuals(channel_names=lead_names, n_cols=4,
                   width=5.9, height=3.5, dpi=300, save_path="Results/Figures/ECGresiduals.png")

# CIs:
alpha_ci, omega_ci, delta_ci, gamma_ci = res.conf_intervals(0.95, method=2)
print(alpha_ci)
res.show_conf_intervals()

############################## CASE 2 ##############################

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


