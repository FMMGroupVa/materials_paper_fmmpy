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

###############################################################################

def beta0(a1, a2):
    return (a1 - a2) / (np.conj(a1) - np.conj(a2))

def beta1(a1, a2):
    return (1 - np.conj(a1) * a2) / (np.conj(a1) - np.conj(a2))

def betaMatrix(an):
    N = len(an)
    beta0Mat = np.zeros((N, N))
    beta1Mat = np.zeros((N, N))
    
    for i in range(N-1):
        for j in range(i+1, N):
            beta0Mat[i, j] = beta0(an[i], an[j])
            beta1Mat[i, j] = beta1(an[i], an[j])
            beta1Mat[j, i] = beta1(an[j], an[i])  # Beta2(i,j) = Beta1(j,i)
    
    beta0Mat = beta0Mat + beta0Mat.T
    return beta0Mat, beta1Mat

def phiMatrix(an):
    beta0Mat, beta1Mat = betaMatrix(an)
    N = len(an)
    phiMat = np.zeros((N, N+1))
    phiMat[0, 1] = 1  # Case k = 1, phi0 = 0, phi1 = 1

    for k in range(1, N):
        prevPhis = phiMat[k-1, 1:k]  # Vector prevPhis: phi_1(k-1), ..., phi_k-1(k-1)
        phiMat[k-1, 0] = np.dot(prevPhis, beta0Mat[0:k-1, k-1])  # phi_0(k)
        phiMat[k-1, 1:(k-1)] = beta1Mat[0:k-1, k-1] * prevPhis  # phi_1(k), ..., phi_k-1(k)
        phiMat[k-1, k] = phiMat[k-2, 0] + np.dot(prevPhis, beta1Mat[k-1, 0:(k-1)])

    phiMat = np.vstack((np.zeros(N+1), phiMat))
    phiMat[0, 0] = 1
    return phiMat

