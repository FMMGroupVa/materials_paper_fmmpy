# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:54:08 2025

@author: Christian
"""

import cProfile
import numpy as np
import pandas as pd
import scipy.signal as sc
import matplotlib.pyplot as plt
import os
os.chdir(r'C:\Users\Christian\Documents\GitHub\PaquetePython')

import fit_fmm
# from  fit_fmm_unit import fit_fmm_unit
from auxiliar_functions import seq_times

#%% EJEMPLO 1
# df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\Patient1.csv', header=None)
# df = df.iloc[:,350:850]

df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\ICARE_0357.csv')
df = df.transpose()

time_points = np.linspace(0, 2 * np.pi, num=df.shape[1]+1)[:-1]

analytic_data_matrix = sc.hilbert(df, axis = 1)
n_ch, n_obs = analytic_data_matrix.shape


omega_grid = np.exp(np.linspace(np.log(0.001), np.log(0.25), 50))
omega_grid = np.reshape(omega_grid, (50))

#%% COMPROBACION DEL AJUSTE 

from fit_fmm_k import fit_fmm_k
from auxiliar_functions import predict, seq_times

time_points = seq_times(n_obs)

n_back = 15

cProfile.run('a_model, coefs_model, phis, prediction = fit_fmm_k(analytic_data_matrix=analytic_data_matrix, n_back=n_back, max_iter=1,time_points=time_points, omega_grid=omega_grid,weights=np.ones(n_ch), post_optimize=False)')

cProfile.run('a_model, coefs_model, phis, prediction = fit_fmm_k(analytic_data_matrix=analytic_data_matrix, n_back=n_back, max_iter=2,time_points=time_points, omega_grid=omega_grid,weights=np.ones(n_ch), post_optimize=False)')









