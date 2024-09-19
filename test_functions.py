# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:08:15 2024

@author: Christian
"""

import numpy as np
import pandas as pd
import scipy.signal as sc
import matplotlib.pyplot as plt
import os
os.chdir(r'C:\Users\Christian\Documents\GitHub\PaquetePython')

import fit_fmm
# from  fit_fmm_unit import fit_fmm_unit
from auxiliar_functions import seq_times

#%%
df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\Patient1.csv')
df = df.iloc[:,350:850]
time_points = np.linspace(0, 2 * np.pi, num=df.shape[1]+1)[:-1]

analytic_data_matrix = sc.hilbert(df, axis = 1)

#%%
'''
plt.plot(time_points, analytic_data_matrix[4].real, color='blue')
plt.plot(time_points, analytic_data_matrix[4].imag, color='red')
plt.show()


# %%

fit_fmm.fit_fmm(df) # Dataframe
fit_fmm.fit_fmm(df.to_numpy()) # ndarray

#%%
omega_grid = np.linspace(0.01,0.99,num=10)

time_points = seq_times(100)

aux2 = np.meshgrid(omega_grid, time_points)

aux3 = (1-aux2[0])/(1+aux2[0])*np.exp(1j*(aux2[1]+np.pi))


plt.plot(aux3.real, aux3.imag, 'o', color='blue')
plt.show()

'''
#%%
from fit_fmm_unit import fit_fmm_unit, szego

#%%
# omega_grid = np.array([0.05, 0.1, 0.15, 0.2, 0.5, 1])
# omega_grid = (1-afdcal.dic_an[0])/(1+afdcal.dic_an[0])*np.exp(0*1j)

# omega_grid = (1-afdcal.dic_an[0])/(1+afdcal.dic_an[0])*np.exp(0*1j)
afd_abs = np.linspace(0, 1, 50, False)
omega_grid = (1-afd_abs)/(1+afd_abs)*np.exp(0*1j)
omega_grid = np.reshape(omega_grid, (50))

c0 = np.zeros(11, dtype=complex)
remainder = np.copy(analytic_data_matrix)
for i in range(df.shape[0]):
    c0[i] = np.mean(analytic_data_matrix[i,:])
    remainder[i,:] = (analytic_data_matrix[i,:] - c0[i])/np.exp(1j*time_points)

aux4 = fit_fmm_unit(analytic_data_matrix = remainder, 
                    time_points = time_points, omega_grid = omega_grid,
                    weights=np.ones(11))

# CHECK - VERSIÓN SIN OPTIM: SALE IGUAL QUE AFD
# an = (0.92677 -0.25041j)

unit_a = aux4.x[0]+1j*aux4.x[1]




















