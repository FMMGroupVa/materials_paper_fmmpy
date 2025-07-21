# -*- coding: utf-8 -*-
import os
os.chdir(r"C:\Users\Christian\Documents\GitHub\PaquetePython")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from PyFMM.fit_fmm import fit_fmm
from scipy.interpolate import interp1d
import time
import sys

if len(sys.argv) > 1:
    N_REPEATS = int(sys.argv[1])
else:
    N_REPEATS = 100
    
# ====== Data: ECG beat (Could be simulated) ======
df = pd.read_csv(r'Data\ECG_data.csv', header=None)
df_base = df.iloc[:, 350:850]  # 500 obs

# ====== Test configurations ======

# 1. Data variations
channels_values = [i+1 for i in range(12)]
# Different sizes under downsampling or interpolation
n_obs_values = [250, 500, 750, 1000] 

# 2. Fitting options variations
n_back_values = [2, 3, 4, 5]
max_iter_values = [1, 5, 10, 15, 20]
post_optimize_values = [True, False]

N_REPEATS = 100

# ========= Store results =========
results = []

for n_obs in n_obs_values:
    if n_obs <= df_base.shape[1]:
        df_obs = df_base.iloc[:, :n_obs]
    else:
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
            print(f"Run: channels={channels}, n_back={n_back}")
            for max_iter in max_iter_values:
                for post_optimize in post_optimize_values:
                    times = []
                    for repeat in range(N_REPEATS):
                        start = time.perf_counter()
                        res = fit_fmm(
                            data_matrix=data_subset,
                            n_back=n_back,
                            max_iter=max_iter,
                            post_optimize=post_optimize,
                            omega_min=0.01,
                            omega_max=0.5
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
results_df.to_csv('timing_results_repeated_nobs.csv', index=False)

# ========= Plot Generation =========
results_df = pd.read_csv('timing_results_repeated_nobs.csv')

# ========= Plot 1 =========
df_plot = results_df[
    (results_df['n_back'] == 5) &
    (results_df['max_iter'] == 10) &
    (results_df['post_optimize'] == True)
]

grouped = df_plot.groupby(['channels', 'n_obs']).agg(
    time_mean=('time_mean', 'mean'),
    time_min=('time_min', 'min'),
    time_max=('time_max', 'max')
).reset_index()

palette = sns.color_palette("tab10", n_colors=len(grouped['n_obs'].unique()))
obs_list = sorted(grouped['n_obs'].unique())

plt.figure(figsize=(5.5, 4.5))

for color, n_obs in zip(palette, obs_list):
    group = grouped[grouped['n_obs'] == n_obs]
    plt.plot(
        group['channels'],
        group['time_mean'],
        marker='o',
        label=f'{n_obs}',
        color=color
    )
    plt.fill_between(
        group['channels'],
        group['time_min'],
        group['time_max'],
        color=color,
        alpha=0.2  # Transparencia baja para no tapar la línea
    )

plt.xlabel('Number of channels', fontsize=12)
plt.ylabel('Execution time (s)', fontsize=12)
plt.legend(title='Number of observations')
plt.grid(True)
plt.savefig('Results/Figures/times1.pdf', dpi=300, transparent=True, bbox_inches='tight')

# Filtro base
df_plot2 = results_df[
    (results_df['channels'] == 4) &
    (results_df['n_obs'] == 500)
]

grouped = df_plot2.groupby(['max_iter', 'n_back', 'post_optimize']).agg(
    time_mean=('time_mean', 'mean'),
    time_min=('time_min', 'min'),
    time_max=('time_max', 'max')
).reset_index()

grouped['post_opt_label'] = grouped['post_optimize'].map({True: 'T', False: 'F'})
grouped['n_back_label'] = grouped['n_back'].astype(str)
grouped['group'] = grouped['n_back_label'] + ', post_opt=' + grouped['post_opt_label']

palette = sns.color_palette("tab10", n_colors=grouped['n_back'].nunique())
n_back_list = sorted(grouped['n_back'].unique())

plt.figure(figsize=(5.5, 4.5))

for color, n_back in zip(palette, n_back_list):
    for post_opt in [True, False]:
        linestyle = '-' if post_opt else '--'
        label = f"n_back={n_back}, post_opt={'T' if post_opt else 'F'}"
        
        group = grouped[(grouped['n_back'] == n_back) & (grouped['post_optimize'] == post_opt)]
        if group.empty:
            continue

        plt.plot(
            group['max_iter'],
            group['time_mean'],
            marker='o',
            linestyle=linestyle,
            label=label,
            color=color
        )
        plt.fill_between(
            group['max_iter'],
            group['time_min'],
            group['time_max'],
            color=color,
            alpha=0.2
        )


handles = []

for color, n_back in zip(palette, n_back_list):
    for post_opt in [True, False]:
        linestyle = '-' if post_opt else '--'
        label = f"n_back={n_back}, post_opt={'T' if post_opt else 'F'}"
        
        group = grouped[(grouped['n_back'] == n_back) & (grouped['post_optimize'] == post_opt)]
        if group.empty:
            continue

        plt.plot(
            group['max_iter'],
            group['time_mean'],
            marker='o',
            linestyle=linestyle,
            label='_nolegend_',
            color=color
        )
        plt.fill_between(
            group['max_iter'],
            group['time_min'],
            group['time_max'],
            color=color,
            alpha=0.2
        )
        handles.append(Line2D(
            [0], [0],
            color=color,
            linestyle=linestyle,
            marker='o',
            label=label
        ))
        
plt.xlabel('Number of iterations (max_iter)', fontsize=12)
plt.ylabel('Execution time (s)', fontsize=12)
plt.xticks([1, 5, 10, 15, 20])
plt.legend(
    handles=handles,
    markerscale=1.3,   # agranda el punto
    handlelength=2.5,    # agranda el segmento de línea
    fontsize=11
)
plt.grid(True)

plt.savefig('Results/Figures/times2.pdf', bbox_inches='tight')
plt.show()

