# -*- coding: utf-8 -*-

import numpy as np
from numba import jit

@jit
def szego(a, t): 
    return ((1 - np.abs(a)**2) ** 0.5) / (1 - np.conj(a)*np.exp(1j*t))

@jit
def mobius(a, t): 
    return ((np.exp(1j*t) - a)) / (1 - np.conj(a)*np.exp(1j*t))

def seq_times(nObs):
    return np.reshape(np.linspace(0, 2 * np.pi, num=nObs+1)[:-1], (1,nObs))

def predict(a, coefs, time_points):
    n_ch, n_coefs = coefs.shape
    n_obs = time_points.shape[0]

    prediction = np.zeros((n_ch, n_obs), dtype = complex)
    blaschke = np.ones((1, n_obs))

    for k in range(n_coefs):
        for ch_i in range(n_ch):
            prediction[ch_i] = prediction[ch_i] + coefs[ch_i,k]*szego(a[k], time_points)*blaschke
        blaschke = blaschke*mobius(a[k], time_points)
    return prediction