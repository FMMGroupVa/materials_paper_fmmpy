# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

from fit_fmm import fit_fmm
from scipy.interpolate import interp1d
import time

#%% Data: ECG beat 
df = pd.read_csv(r'Patient1.csv', header=None)
df_base = df.iloc[:, 350:850]  # 500 obs


res = fit_fmm(data_matrix=df, # Data
              n_back=5, max_iter=10, post_optimize=True,  # Fit options
              omega_min=0.01, omega_max=0.5) # Parameter control

#%% Data: ECG beat 

# === Test configurations ===
channels_values = [i+1 for i in range(12)]
n_back_values = [2, 3, 4, 5]
max_iter_values = [1, 5, 10, 15, 20]
post_optimize_values = [True, False]
n_obs_values = [250, 500, 750, 1000]

# === Repeticiones ===
N_REPEATS = 5

# === Store results ===
results = []

for n_obs in n_obs_values:
    if n_obs <= 500:
        df_obs = df_base.iloc[:, :n_obs]
    else:
        # Interpolar para aumentar longitud
        original_time = np.linspace(0, 1, 500)
        new_time = np.linspace(0, 1, n_obs)
        interpolated = []

        for idx in range(df_base.shape[0]):
            f = interp1d(original_time, df_base.iloc[idx, :], kind='cubic')
            interpolated.append(f(new_time))

        df_obs = pd.DataFrame(interpolated)

    for channels in channels_values:
        data_subset = df_obs.iloc[0:channels, :]

        for n_back in n_back_values:
            for max_iter in max_iter_values:
                for post_optimize in post_optimize_values:
                    
                    times = []
                    for repeat in range(N_REPEATS):
                        print(f"Run: channels={channels}, n_back={n_back}, "
                              f"max_iter={max_iter}, post_optimize={post_optimize}, "
                              f"n_obs={n_obs} [Repeat {repeat + 1}/{N_REPEATS}]")

                        start = time.perf_counter()
                        res = fit_fmm(
                            data_matrix=data_subset,
                            n_back=n_back,
                            max_iter=max_iter,
                            post_optimize=post_optimize,
                            omega_min=0.01,
                            omega_max=0.5,
                            verbose=False
                        )
                        end = time.perf_counter()
                        elapsed = end - start
                        times.append(elapsed)

                    results.append({
                        'channels': channels,
                        'n_back': n_back,
                        'max_iter': max_iter,
                        'post_optimize': post_optimize,
                        'n_obs': n_obs,
                        'time_mean': np.mean(times),
                        'time_std': np.std(times),
                        'time_min': np.min(times),
                        'time_max': np.max(times),
                        'n_repeats': N_REPEATS
                    })

# === Save ===
results_df = pd.DataFrame(results)
print("\nTiming summary:")
print(results_df)

results_df.to_csv('timing_results_repeated_nobs.csv', index=False)