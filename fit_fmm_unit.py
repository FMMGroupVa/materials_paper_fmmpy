# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:07:59 2024

@author: Christian
"""
import numpy as np
import numpy.matlib as pymat
from numpy.fft import fft, ifft
from scipy.optimize import minimize

# 
def szego(a, t): 
    return ((1 - np.abs(a)**2) ** 0.5) / (1 - np.conj(a)*np.exp(1j*t))

# Funcion a minimizar (cuidado que queremos el máximo de abs(coef),
# usamos el signo contrario)
# Los pesos no son para el inner product, son para ponderar los cuadrados de 
# las normas a maximizar. Esto es algo "FMM", las expresiones de la  
# verosimilitud están ponderadas con la desviación típica de los residuales, 
# tengo que comprobar si puedo calcularlos con normas AFD o "traducir a FMM"
def inner_products_sum(splitted_a, analytic_data_matrix, t, weights):
    a = splitted_a[0]+1j*splitted_a[1]
    if abs(a)>=1:
        return float('inf')
    
    sum_abs = 0
    for i_ch in range(analytic_data_matrix.shape[0]):
        sum_abs = sum_abs + weights[i_ch]*(
            abs(np.conj(szego(a, t).dot(
                analytic_data_matrix[i_ch,:].conj().T)
                )) ** 2
            )
    return -sum_abs

# Dividimos un complejo en real/imaginario para poder pasarlo al optimizador.
def split_complex(z): 
    return ((z.real, z.imag))

# Argumentos, ¿pasar el omega grid directamente?
def fit_fmm_unit(analytic_data_matrix, time_points=None,
                 # omega_min=0.001, omega_max = 0.99, lengthOmegaGrid=24, 
                 omega_grid=None, weights=None):
    
    # Nota: en los códigos del AFD guardan cada estructura como si fuera 
    # distinta en cada canal (lo repiten n_ch veces). Los time_points los 
    # asumimos iguales en cada canal y también el grid, la base para fft...
    
    n_ch, n_obs = analytic_data_matrix.shape
    fmm_grid = np.meshgrid(omega_grid, time_points)
    afd_grid = (1-fmm_grid[0])/(1+fmm_grid[0])*np.exp(1j*(fmm_grid[1]))

    modules_grid = (1-omega_grid)/(1+omega_grid)*np.exp(1j*0)
    an_search_len = modules_grid.shape[0]
    
    # BREVE EXPLICACIÓN SOBRE LA BASE:

    # Las componentes AFD son c_k*B_k(t), 
    # donde B_k(t) = e(t, a[k])*m(a[1],t)*...*m(a[k-1],t).
    
    # No se trabaja con B_k(t), si no que en k>1 se multiplica por 1/m(a[1],t)*
    # ...*1/m(a[k-1],t) para siempre manejar e(t, a[k]).
    # Es por eso que la base se calcula y almacena 1 sola vez
    
    # Lo que llamamos base es la DFT (Discrete Fourier Transform) de los 
    # e(t, a_i) para cada a_i que están en la recta real positiva 
    # (traducción FMM: varían los omegas y alpha=0, creo que se podría fijar 
    # cualquier valor de alpha).
    # Luego se aplica la DFT sobre los datos X, se repite tantas veces como 
    # valores de omega tengamos y se multiplica por la base. Luego revertimos,
    # y aplicamos la transformada inversa (IDFT).
    
    # NOTA: si hacemos una función para cada ajuste individual (esta función 
    # fit_fmm_unit(...), de hecho), hay que calcular la base otra vez o pasarla 
    # como argumento, previsiblemente más tiempo que solo acceder a un atributo 
    # como ellos. (En python no hay punteros a memoria que yo sepa...).
    
    base = np.zeros((modules_grid.shape[0], n_obs), dtype=complex)
    for i in range(an_search_len):
        base[i,:] = fft(szego(modules_grid[i], time_points), n_obs)
    
    # IMPORTANTE:
        
    # He comparado la base con la que sale en el AFD, los coeficiente mas significativos
    # son iguales y luego parece que se arrastra algo de error numerico... No deberia
    # afectar mucho, pero es extraño aún así.
    
    abs_coefs = 0
    # Paso del grid, nodos aproximados por fft
    for ch_i in range(n_ch):
        #abs_coefs += np.abs(ifft(pymat.repmat(fft(analytic_data_matrix[ch_i, :], n_obs), an_search_len, 1) * base, n_obs, 1))
        abs_coefs += np.abs(ifft(np.repeat(fft(analytic_data_matrix[ch_i, :], n_obs)[np.newaxis, :], an_search_len, axis=0) * base, n_obs, 1))
    
    abs_coefs = abs_coefs.T
    
    max_loc_tmp = np.argwhere(abs_coefs == np.amax(abs_coefs))
    best_a = afd_grid[max_loc_tmp[0, 0], max_loc_tmp[0, 1]]
    
    # -------------------- DEVELOPING --------------------

    res = minimize(inner_products_sum, x0=split_complex(best_a), 
                   args=(analytic_data_matrix, time_points, weights), 
                   method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    
    # ------------------ END DEVELOPING ------------------
    
    return res
    



    
    
