import numpy as np
import pandas as pd
import time
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from PyFMM.fit_fmm import fit_fmm

import sys

# Si se pasa un argumento, úsalo. Si no, usa 100 por defecto:
if len(sys.argv) > 1:
    N_REPEATS = int(sys.argv[1])
else:
    N_REPEATS = 5

print(f"N_REPEATS set to: {N_REPEATS}")


df_raw = pd.read_csv('Data/XPS_Fe2p_data.csv')
x = df_raw.iloc[:, 0]  
df_signal = df_raw.iloc[:, 1:5].T

# === Config general ===
channels_fixed = 4
results = []

# === Experimento 1: Escalabilidad n_obs ===
max_iter = 5
n_back = 5
post_opt = True

channels_values = [1, 2, 3, 4, 5, 6, 7, 8]   # Ahora hasta 8
n_obs_values = [200, 300, 400, 500]
N_REPEATS = 5  
results = []

for n_obs in n_obs_values:
    # Interpolación si hace falta
    orig_time = np.linspace(0, 1, df_signal.shape[1])
    new_time = np.linspace(0, 1, n_obs)
    interpolated = []
    for idx in range(df_signal.shape[0]):
        f = interp1d(orig_time, df_signal.iloc[idx, :], kind='cubic')
        interpolated.append(f(new_time))
    df_obs = pd.DataFrame(interpolated)

    for channels in channels_values:
        # Repetir si hace falta
        times_to_repeat = int(np.ceil(channels / df_obs.shape[0]))
        df_expanded = pd.concat([df_obs] * times_to_repeat, ignore_index=True)
        df_subset = df_expanded.iloc[0:channels, :]

        print(f'Run: channels={channels}, n_obs={n_obs}')
        
        times = []
        for _ in range(N_REPEATS):
            start = time.perf_counter()
            res = fit_fmm(
                data_matrix=df_subset,
                n_back=n_back,
                max_iter=max_iter,
                post_optimize=post_opt,
                omega_min=0.01,
                omega_max=0.1,
                length_alpha_grid=50,
                beta_min=np.pi - 0.2,
                beta_max=np.pi + 0.2
            )
            end = time.perf_counter()
            times.append(end - start)
            print(end - start)
        results.append({
            'channels': channels,
            'n_obs': n_obs,
            'n_back': n_back,
            'max_iter': max_iter,
            'post_optimize': True,
            'time_mean': np.mean(times),
            'time_min': np.min(times),
            'time_max': np.max(times),
            'exp': 'obs_scaling'
        })

# === Experiment 2: max_iter, n_back, post_opt ===

n_back_values = [2, 3, 4, 5]
max_iter_values = [1, 5, 10, 15, 20]
post_optimize_values = [True, False]

for n_back in n_back_values:
    for max_iter in max_iter_values:
        for post_opt in post_optimize_values:
            times = []
            for _ in range(N_REPEATS):
                start = time.perf_counter()
                res = fit_fmm(
                    data_matrix=df_signal.iloc[0, :],
                    n_back=n_back,
                    max_iter=max_iter,
                    post_optimize=post_opt,
                    omega_min=0.01,
                    omega_max=0.1,
                    length_alpha_grid=50,
                    beta_min=np.pi - 0.25,
                    beta_max=np.pi + 0.25
                )
                end = time.perf_counter()
                times.append(end - start)

            results.append({
                'exp': 'iter_nback_scaling',
                'channels': 1,
                'n_obs': df_signal.shape[1],
                'n_back': n_back,
                'max_iter': max_iter,
                'post_optimize': post_opt,
                'time_mean': np.mean(times),
                'time_max': np.max(times),
                'n_repeats': N_REPEATS
            })

# === Guardar ===
df_results = pd.DataFrame(results)
df_results.to_csv('Results/timing_XPS_both_experiments.csv', index=False)

results_df = pd.read_csv('Results/timing_XPS_both_experiments.csv')

# === Filtrar SOLO las combinaciones del primer experimento ===
df_plot = results_df[
    results_df['exp'] == 'obs_scaling'
]

# === Agrupar y calcular ===
grouped = df_plot.groupby(['channels', 'n_obs']).agg(
    time_mean=('time_mean', 'mean'),
    time_min=('time_min', 'min'),
    time_max=('time_max', 'max')
).reset_index()

# === Paleta por número de obs ===
palette = sns.color_palette("tab10", n_colors=len(grouped['n_obs'].unique()))
obs_list = sorted(grouped['n_obs'].unique())

plt.figure(figsize=(5.5, 4.5))

for color, n_obs in zip(palette, obs_list):
    group = grouped[grouped['n_obs'] == n_obs]
    plt.plot(
        group['channels'],
        group['time_mean'],
        marker='o',
        label=f'{n_obs} obs',
        color=color
    )
    plt.fill_between(
        group['channels'],
        group['time_min'],
        group['time_max'],
        color=color,
        alpha=0.2
    )

plt.xlabel('Number of channels', fontsize=12)
plt.ylabel('Execution time (s)', fontsize=12)
plt.title('Scalability with signal length and channels')
plt.legend(title='Number of obs')
plt.grid(True)
plt.tight_layout()
plt.savefig('Results/Figures/XPS_times1.pdf', dpi=300, transparent=True, bbox_inches='tight')
plt.show()


# === Leer resultados ===
results_df = pd.read_csv('Results/timing_XPS_both_experiments.csv')

# === Filtrar SOLO segundo experimento ===
df_plot2 = results_df[
    results_df['exp'] == 'iter_nback_scaling'
]

# === Agrupar ===
grouped = df_plot2.groupby(['max_iter', 'n_back', 'post_optimize']).agg(
    time_mean=('time_mean', 'mean'),
    time_min=('time_mean', 'min'),
    time_max=('time_mean', 'max')
).reset_index()

# Etiquetas para leyenda
grouped['post_opt_label'] = grouped['post_optimize'].map({True: 'T', False: 'F'})
grouped['group'] = 'n_back=' + grouped['n_back'].astype(str) + ', post_opt=' + grouped['post_opt_label']

# Paleta: un color por n_back
palette = sns.color_palette("tab10", n_colors=grouped['n_back'].nunique())
n_back_list = sorted(grouped['n_back'].unique())

plt.figure(figsize=(5.5, 4.5))

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
plt.title('Scalability: iterations, n_back, post_opt')
plt.xticks([1, 5, 10, 15, 20])
plt.legend(    handles=handles,
    markerscale=1.3,   # agranda el punto
    handlelength=2.5,    # agranda el segmento de línea
    fontsize=11)
plt.grid(True)
plt.tight_layout()
plt.savefig('Results/Figures/XPS_times2.pdf', dpi=300, transparent=True, bbox_inches='tight')
plt.show()

