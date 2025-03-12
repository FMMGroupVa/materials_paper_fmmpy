# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:17:22 2025

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

#%% 

#%% COMPROBACION DEL AJUSTE 

from fit_fmm_k import fit_fmm_k
from auxiliar_functions import predict, seq_times

time_points = seq_times(n_obs)

n_back = 15
a_model, coefs_model, phis, prediction = fit_fmm_k(
    analytic_data_matrix=analytic_data_matrix, n_back=n_back, max_iter=20,
    time_points=time_points, omega_grid=omega_grid,
    weights=np.ones(n_ch), post_optimize=False)

prediction2 = predict(a_model, coefs_model, time_points)

#%%

a1 = np.array([ 0.  +0.j  ,  0.22+0.82j, -0.75+0.41j, -0.73-0.31j,  0.55-0.64j,
                0.23+0.77j,  0.35-0.8j ,  0.29-0.92j, -0.84+0.12j, -0.75+0.23j,
                0.78+0.44j, -0.68-0.56j, -0.69+0.47j,  0.79+0.11j,  0.57+0.77j,
                0.37+0.75j])

c1 = np.array([-0.61 +0.j,  37.45+60.41j,  114.25+59.43j, -165.87-68.01j, -12.38-48.94j,    
               4.62 +1.21j,  57.69-97.64j,  1.66 +3.05j,  13.13-45.91j,  40.17 -6.07j,   
               44.62 -5.62j, -79.55-61.21j, -45.14-29.95j, -21.64+23.64j, 61.22+37.41j,  
               -21.93 +6.46j])

#%%
(1-np.abs(a1)**2)/(1+np.abs(a1)**2)
#%%
np.angle(a1)  +np.pi
#%%
prediction2 = predict(a1, coefs_model, time_points)
plt.plot(time_points[0], prediction2[0].real, color='red')


#%%
# pd.DataFrame(prediction2.real[0]).to_csv('Example1.csv', index=False)

#%%
model1 = prediction2[0]

max_comps = 15

R2j = np.zeros(max_comps)
for j in range(1,max_comps+1):
    a, coefs, phis, prediction = fit_fmm_k(
        analytic_data_matrix=model1, n_back=j, max_iter=1,
        time_points=time_points, omega_grid=omega_grid,
        weights=np.ones(n_ch), post_optimize=False)
    
    plt.plot(time_points[0], model1.real, color='blue')
    plt.plot(time_points[0], prediction[0].real, color='red')
    plt.show()
    
    R2j[j-1] = 1-np.var(model1.real-prediction.real[0])/np.var(model1.real)
    
# pd.DataFrame(prediction.real[0]).to_csv('Example1_AFD.csv', index=False)



#%%
from scipy.io import loadmat

# pat = 316
# hour = 29

pat = 359
hour = 22

# filemat = r"C:\Users\Christian\Christian\ChallengePhysionet23\codesMATLAB\training_subset\ICARE_0"+str(pat)+"\ICARE_0"+str(pat)+"_"+str(hour)+".mat"
filemat = r"C:\Users\Christian\Christian\ChallengePhysionet23\codesMATLAB\training_subset\ICARE_0"+str(pat)+"\ICARE_0"+str(pat)+"_"+str(hour)+".mat"

data = loadmat(filemat)['val']
data2 = data[:,22000:23500]
row_mean = data2.mean(axis=1, keepdims=True)
row_std = data2.std(axis=1, keepdims=True)
# Standardize each row
data2 = (data2 - row_mean) / row_std

plt.plot(data2[0], color='blue')
#%%
from fit_fmm_k import fit_fmm_k
from auxiliar_functions import predict, seq_times

analytic_data_matrix = sc.hilbert(data2, axis = 1)
n_ch, n_obs = analytic_data_matrix.shape

omega_grid = np.exp(np.linspace(np.log(0.001), np.log(0.25), 50))
omega_grid = np.reshape(omega_grid, (50))

time_points = seq_times(n_obs)

n_back = 15
a_model, coefs_model, phis, prediction = fit_fmm_k(
    analytic_data_matrix=analytic_data_matrix[0], n_back=n_back, max_iter=40,
    time_points=time_points, omega_grid=omega_grid,
    weights=np.ones(n_ch), post_optimize=False)

prediction2 = predict(a_model, coefs_model, seq_times(n_obs))

#%%
plt.plot(time_points[0], data2[0], color='blue')
plt.plot(time_points[0], prediction2[0].real, color='red')

(1-np.abs(a_model)**2)/(1+np.abs(a_model)**2)

#%%

model1 = predict(a_model, coefs_model, seq_times(n_obs))[0]
# pd.DataFrame(model1.real).to_csv(r'C:\Users\Christian\Christian\FMM-Python\FMM\ExampleCristina.csv', index=False)

max_comps = 15

R2j = np.zeros(max_comps)
for j in range(1,max_comps+1):
    a, coefs, phis, prediction = fit_fmm_k(
        analytic_data_matrix=model1, n_back=j, max_iter=1,
        time_points=time_points, omega_grid=omega_grid,
        weights=np.ones(n_ch), post_optimize=False)
    
    plt.plot(time_points[0], model1.real, color='blue')
    plt.plot(time_points[0], prediction[0].real, color='red')
    plt.show()
    
    R2j[j-1] = 1-np.var(model1.real-prediction.real[0])/np.var(model1.real)
    
# pd.DataFrame(prediction.real[0]).to_csv(r'C:\Users\Christian\Christian\FMM-Python\FMM\ExampleCristina_AFD.csv', index=False)
