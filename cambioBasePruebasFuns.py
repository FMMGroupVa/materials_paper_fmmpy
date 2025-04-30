# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 18:43:28 2025

@author: Christian
"""

import numpy as np

def beta0(a1, a2):
    return (a1 - a2) / (np.conj(a1) - np.conj(a2))

def beta1(a1, a2):
    return (1 - np.conj(a1) * a2) / (np.conj(a1) - np.conj(a2))

def beta_matrix(an):
    N = len(an)
    beta0Mat = np.zeros((N, N), dtype=complex)
    beta1Mat = np.zeros((N, N), dtype=complex)
    
    for i in range(N-1):
        for j in range(i+1, N):
            beta0Mat[i, j] = beta0(an[i], an[j])
            beta1Mat[i, j] = beta1(an[i], an[j])
            beta1Mat[j, i] = beta1(an[j], an[i])  # Beta2(i,j) = Beta1(j,i)
    
    beta0Mat = beta0Mat + beta0Mat.T
    return beta0Mat, beta1Mat

def calculate_xi_matrix(an):
    beta0_mat, beta1_mat = beta_matrix(an)
    N = len(an)
    
    # Initialize xi_mat with an extra row of zeros at the end (size (N+1, N+1))
    xi_mat = np.zeros((N + 1, N + 1), dtype=complex)
    
    xi_mat[0, 0] = 1
    xi_mat[1, 1] = 1
    # Loop through k from 2 to N
    for k in range(2, N+1):
        # Vector prev_xis: [xi_1(k-1), ..., xi_k-1(k-1)] (indexing starts at 0)
        prev_xis = xi_mat[k-1, 1:k]

        # Compute xi_0(k) as the dot product of prev_xis and the corresponding beta0_mat column
        xi_mat[k, 0] = np.dot(prev_xis, beta0_mat[:k-1, k-1])
        
        # Compute xi_1(k), ..., xi_k-1(k) using element-wise multiplication
        xi_mat[k, 1:k] = beta1_mat[:k-1, k-1] * prev_xis
        
        #print(prev_xis )
        # Compute xi_k(k) as a scalar product of [xi0(k-1), prev_xis] and [1, beta1(1,k), ..., beta1(k-1,k)]
        xi_mat[k, k] = xi_mat[k-1, 0] + np.dot(prev_xis, beta1_mat[k-1, :k-1])
        #print(xi_mat[k + 1, k+1])
    
    return xi_mat

def transition_matrix(an):
    """
    Computes the transition matrix M for the change of basis from B to B' where:
    B' is a Möbius basis and B is a Takenaka-Malmquist basis.
    
    Parameters:
    an : numpy array
        A vector of size [K] with parameters a1,...,aK all different.
        
    Returns:
    M : numpy array
        Transition matrix of dimension (K+1)x(K+1).
    """
    K = len(an)-1  # Length of the an vector
    M = np.zeros((K+1, K+1), dtype=complex)  # Initialize the (K+1)x(K+1) matrix with zeros
    xiMat = calculate_xi_matrix(an[1:])  # Call the xiMatrix function

    # The 1st column 1 is (1,0,...,0)': both bases have an intercept as first element
    M[0, 0] = 1

    # Loop through each k for the first column and the upper triangular part
    for k in range(K):
        # c0 coefficient (first row, element by element)
        M[0, k+1] = (an[k+1] * xiMat[k, 0] + xiMat[k+1, 0]) / np.sqrt(1 - np.abs(an[k+1])**2)
        
        # Triangular matrix: only upper triangle is not zero. Each k is a row.
        for j in range(k+1, K):
            
            M[k+1, j+1] = (np.sqrt(1 - np.abs(an[j+1])**2) * xiMat[j, k+1] /
                           (np.conj(an[k+1]) - np.conj(an[j+1])))

        # Diagonal elements of M
        M[k+1, k+1] = xiMat[k+1, k+1] / np.sqrt(1 - np.abs(an[k+1])**2)

    return M

#%%

a1 = 0.76*np.exp(1j*3)
a2 = 0.69*np.exp(1j*4)
a3 = 0.91*np.exp(1j*4.5)

beta0(a1,a2)
#(-0.6270051177357103-0.7790151361387199j)
#-0.6270 - 0.7790i

beta1(a1,a2)
#(0.1268189132885186+1.1992886854488325j)
#0.1268 + 1.1993i


bm0, bm1 = beta_matrix([a1,a2,a3])


xis = calculate_xi_matrix([a1,a2,a3])

tmat = transition_matrix([0,a1,a2,a3])


























