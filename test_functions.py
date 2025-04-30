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

from fit_fmm import fit_fmm
# from  fit_fmm_unit import fit_fmm_unit
from auxiliar_functions import seq_times

#%%
df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\Patient1.csv', header=None)
df = df.iloc[:,350:850]

time_points = np.linspace(0, 2 * np.pi, num=df.shape[1]+1)[:-1]

analytic_data_matrix = sc.hilbert(df, axis = 1)
n_ch, n_obs = analytic_data_matrix.shape

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
from fit_fmm import fit_fmm

res = fit_fmm(data_matrix=df, n_back=5, max_iter=5, post_optimize=False, 
              omega_min=0.01, omega_max=0.99)

#%%
res.plot_predictions(channels = [0,1,2,3], dpi=300)

#%%
from fit_fmm_unit import fit_fmm_unit

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

# abs(unit_a)
# np.angle(unit_a)

#%% COMPROBACION DEL AJUSTE 

from fit_fmm_k import fit_fmm_k
from auxiliar_functions import predict, predictFMM, seq_times, transition_matrix

time_points = seq_times(500)

n_back = 5
a, coefs, phis, prediction = fit_fmm_k(analytic_data_matrix=analytic_data_matrix, 
                                 n_back=n_back, time_points=time_points, 
                                 omega_grid=omega_grid,
                                 weights=np.ones(n_ch), post_optimize=False)

prediction2 = predict(a, coefs, time_points)


#%% PLOT DATA VS PREDICTION (UN CANAL)

plt.plot(time_points[0], analytic_data_matrix[0].real, color='blue')
plt.plot(time_points[0], prediction2[0].real+coefs[0,0].real, color='red')
plt.show()

#%%
from auxiliar_functions import predict, predictFMM, transition_matrix

AFD_to_FMM_matrix = transition_matrix(a)
phis = np.dot(AFD_to_FMM_matrix, coefs.T)

#%%
yFMM = predictFMM(a[1:], phis[:,1:], seq_times(100))


#%% PROFILING 
from fit_fmm_k import fit_fmm_k
from timeit import timeit
from cProfile import Profile
from pstats import SortKey, Stats
    
#%%

with Profile() as profile:
    fit_fmm_k(analytic_data_matrix=analytic_data_matrix, n_back=10,
              time_points=time_points, omega_grid=omega_grid, 
              weights=np.ones(11), post_optimize=False)
    (Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats())
    
    '''
    Mismos datos, comparacion Codigo1 (AFD), Codigo2 (Nuestro).
    
    Tiempo Codigo1:
    31723 function calls (31648 primitive calls) in 0.128 seconds
    
    Tiempo Codigo2 (sin optim):
    4301 function calls (4270 primitive calls) in 0.048 seconds
    
    Tiempo Codigo2 (con optim):
    102241 function calls (102210 primitive calls) in 0.298 seconds
    
    '''
    
    '''
    Conclusiones:
        - Cuello de botella en el step de optimización
        - usando numba disminuye mucho el tiempo
        - la opción'xatol' no parece afectar al tiempo  
    '''

#%%

from auxiliar_functions import predict, predictFMM, seq_times, transition_matrix, calculate_xi_matrix

N = 5;

rad = np.array([0.7, 0.85, 0.9, 0.45, 0.7]);
alpha = np.array([0.5, np.pi, 2.1, 5, 4.25]);

Cn = np.zeros((1,5), dtype = complex)
an = rad*np.exp(1j*alpha);

Cn = np.zeros((1,6), dtype = complex)
coef_fase = np.array([np.pi, 0, np.pi/2, 0, np.pi/2])-alpha;
Amplitudes = np.array([5, 4, 1, 1, 2.5]);
Cn[0,1:] = Amplitudes*np.exp(1j*coef_fase);
Cn[0,0] = 5;

#%%
an2 = np.insert(an, 0, 0, axis=0)

AFD_to_FMM_matrix = transition_matrix(an2)

phis = np.dot(AFD_to_FMM_matrix, Cn.T).T

xi_mat = calculate_xi_matrix(an)


#%%
yAFD = predict(an2, Cn, seq_times(100))
yFMM = predictFMM(an2, phis, seq_times(100))
plt.plot(seq_times(100)[0,:], yAFD[0,:].real, color='blue')
plt.plot(seq_times(100)[0,:], yFMM[0,:].real, color='red')




