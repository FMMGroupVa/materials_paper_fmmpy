# -*- coding: utf-8 -*-

import os
os.chdir(r"C:\Users\Christian\Documents\GitHub\PaquetePython")

import numpy as np
import pandas as pd
import scipy.signal as sc
from AFDCal import AFDCal
from PyFMM.fit_fmm import fit_fmm
import time
import sys

if len(sys.argv) > 1:
    N_REPEATS = int(sys.argv[1])
else:
    N_REPEATS = 100


df = pd.read_csv(r'Data\ECG_data.csv', header=None)
df_base = df.iloc[:, 350:850]  # 500 obs

def run_afd_decomposition(data_matrix, n_back=5):
    if isinstance(data_matrix, pd.DataFrame):
        data_matrix = data_matrix.values
    afdcal = AFDCal()
    analytic_data_matrix = sc.hilbert(data_matrix, axis=1)
    afdcal.loadInputSignal(analytic_data_matrix)
    afdcal.setDecompMethod(4)
    afdcal.setDicGenMethod(2)
    afdcal.genDic(1/50, 1)
    afdcal.genEva()
    
    afdcal.init_decomp()
    for level in range(n_back):
        afdcal.nextDecomp()

    return afdcal

times_afd = []

for i in range(N_REPEATS):
    start = time.perf_counter()
    run_afd_decomposition(df_base, n_back=5)
    end = time.perf_counter()
    elapsed = end - start
    times_afd.append(elapsed)

mean_time_afd = np.mean(times_afd)
max_time_afd = np.max(times_afd)

# === FMM ===
times_fmm = []

for i in range(N_REPEATS):
    start = time.perf_counter()
    res = fit_fmm(
        data_matrix=df_base,
        n_back=5,
        max_iter=1,
        post_optimize=False,
        length_omega_grid=50,
        omega_min=0.01,
        omega_max=0.5,
        verbose=False
    )
    end = time.perf_counter()
    elapsed = end - start
    times_fmm.append(elapsed)

mean_time_fmm = np.mean(times_fmm)
max_time_fmm = np.max(times_fmm)

# === Save to text file ===
output_text = (
    f"AFD:\n"
    f"Mean execution time over {N_REPEATS} runs: {mean_time_afd:.4f} s\n"
    f"Max time: {max_time_afd:.4f} s\n\n"
    f"FMM:\n"
    f"Mean execution time over {N_REPEATS} runs: {mean_time_fmm:.4f} s\n"
    f"Max time: {max_time_fmm:.4f} s\n"
)

with open("Results/execution_times_FMM_AFD.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

print("\nResults saved to 'Results/execution_times_FMM_AFD.txt'")


P_alpha = (4.2, 5.4)
P_ome = (0.05, 0.25)
QRS_alpha = (5.4, 6.2)
QRS_ome = (0.01, 0.10)
T_alpha = (0, 3.14)
T_ome = (0.1, 0.5)

alpha_restr = np.array([P_alpha, QRS_alpha, QRS_alpha, QRS_alpha, T_alpha])
omega_restr = np.array([P_ome, QRS_ome, QRS_ome, QRS_ome, T_ome])

# === Repeticiones ===
times_restricted = []
times_unrestricted = []

# === Fitting con restricciones ===
for _ in range(N_REPEATS):
    start = time.perf_counter()
    res = fit_fmm(
        data_matrix=df_base,
        n_back=5,
        max_iter=8,
        post_optimize=True,
        alpha_restrictions=alpha_restr,
        omega_restrictions=omega_restr,
        omega_min=0.01,
        omega_max=0.5,
        verbose=False
    )
    end = time.perf_counter()
    times_restricted.append(end - start)

# === Fitting sin restricciones ===
for _ in range(N_REPEATS):
    start = time.perf_counter()
    res = fit_fmm(
        data_matrix=df_base,
        n_back=5,
        max_iter=8,
        post_optimize=True,
        omega_min=0.01,
        omega_max=0.5,
        verbose=False
    )
    end = time.perf_counter()
    times_unrestricted.append(end - start)

# === Estadísticos ===
mean_restricted = np.mean(times_restricted)
max_restricted = np.max(times_restricted)
mean_unrestricted = np.mean(times_unrestricted)
max_unrestricted = np.max(times_unrestricted)

print(f"Restricted mean: {mean_restricted:.4f} s (max: {max_restricted:.4f} s)")
print(f"Unrestricted mean: {mean_unrestricted:.4f} s (max: {max_unrestricted:.4f} s)")

# === Guardar a fichero ===
output_text = (
    f"Restricted:\n"
    f"Mean execution time over {N_REPEATS} runs: {mean_restricted:.4f} s\n"
    f"Max time: {max_restricted:.4f} s\n\n"
    f"Unrestricted:\n"
    f"Mean execution time over {N_REPEATS} runs: {mean_unrestricted:.4f} s\n"
    f"Max time: {max_unrestricted:.4f} s\n"
)

with open("Results/execution_times_restriction_comparison.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

print("\nResults saved to 'Results/execution_times_restriction_comparison.txt'")

