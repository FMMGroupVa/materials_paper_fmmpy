# -*- coding: utf-8 -*-
"""
@author: Christian
"""

import numpy as np
import pandas as pd
from numpy.fft import fft, ifft
from scipy.optimize import minimize, Bounds
from numba import jit
import matplotlib.pyplot as plt
from auxiliar_functions import seq_times, szego, mobius, predict, predict2, transition_matrix, inner_products_sum_2, split_complex
from qpsolvers import solve_ls, solve_qp

def generate_G(p, a, b):
    G = np.zeros((2*p, 2*p+1))
    for var in range(p):
        G[2*var, 2*var+1] = np.sin(a)
        G[2*var, 2*var+2] = np.cos(a)
        G[2*var+1, 2*var+1] = -np.sin(b)
        G[2*var+1, 2*var+2] = -np.cos(b)
    return G

df = pd.read_csv(r'C:\Users\Christian\Documents\GitHub\PaquetePython\example_Fe2p.csv')
df = df.iloc[:,1:5].T
import scipy.signal as sc
analytic_data_matrix = sc.hilbert(df, axis = 1)
data_matrix = analytic_data_matrix.real

# a = np.array([0.+0.j,  
#               0.091115  -0.86093484j,  
#               0.62488678-0.62411581j,-0.14142367-0.43487863j,  
#               0.26917202-0.88078132j])

a = np.array([0.        +0.j        , 0.54682438-0.64607446j])

n_ch, n_obs = df.shape
time_points = seq_times(n_obs)
beta_min = np.pi-0.5
beta_max = np.pi+0.5





n_back = len(a) - 1
n_ch, n_obs = data_matrix.shape

# 1. AFD to Complex FMM
G = generate_G(n_back, beta_min, beta_max)
h = np.zeros(2 * n_back)

# 2. Design matrix 
alphas = np.angle(a[1:]) + np.pi
omegas = (1 - np.abs(a[1:])) / (1 + np.abs(a[1:]))

ts = [2*np.arctan(omegas[i] * np.tan((time_points[0] - alphas[i])/2)) for i in range(n_back)]
DM = np.column_stack([np.ones(n_obs)] + [np.column_stack([np.cos(ts[i]), np.sin(ts[i])]) for i in range(n_back)])

# 4. Allocate storage
RLS = np.zeros((n_ch, 2 * n_back + 1))
phis = np.zeros((n_ch, n_back + 1), dtype=np.complex128)

# 5. Solve LSQ problem for all channels
for ch_i in range(n_ch):
    # RLS[ch_i] = solve_ls(DM, data_matrix[ch_i], G=G, h=h, solver='quadprog')
    RLS[ch_i] = solve_qp(DM.T@DM, -DM.T@data_matrix[ch_i], G=G, h=h, solver='quadprog')

# 6. Compute betas, amplitudes, and phis using vectorized operations
betas = np.arctan2(-RLS[:, 2::2], RLS[:, 1::2]) % (2*np.pi)
amplitudes = np.sqrt(RLS[:, 1::2] ** 2 + RLS[:, 2::2] ** 2)

phis[:, 0] = RLS[:, 0]
phis[:, 1:] = amplitudes * np.exp(1j * (betas - alphas))

coefs2 = np.dot(np.linalg.inv(transition_matrix(a)), phis.T).T

phis2 = np.dot(transition_matrix(a), coefs2.T).T

prediction = np.dot(DM, RLS.T)

res_sq = (data_matrix - prediction.T)**2

rss_ch = np.sum(res_sq, axis=1)  # shape (4,)








