# -*- coding: utf-8 -*-
import os
os.chdir(r"C:\Users\Christian\Documents\GitHub\PaquetePython")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PyFMM.fit_fmm import fit_fmm
from scipy.interpolate import interp1d
import time

#%% Data: ECG beat 
df = pd.read_csv(r'Patient1.csv', header=None)
df_base = df.iloc[:, 350:850]  # 500 obs

# res = fit_fmm(data_matrix=df_base, # Data
#               n_back=5, max_iter=10, post_optimize=True,  # Fit options
#               omega_min=0.01, omega_max=0.5) # Parameter control

#%% Data: ECG beat 

# === Test configurations ===
channels_values = [i+1 for i in range(12)]
n_back_values = [2, 3, 4, 5]
max_iter_values = [1, 5, 10, 15, 20]
post_optimize_values = [True, False]
n_obs_values = [250, 500, 750, 1000]

# === Repeticiones ===
N_REPEATS = 100

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


#%% Correccion (BORRAR)

# Carga resultados anteriores
results_df = pd.read_csv('timing_results_repeated_nobs.csv')

# === Identifica qué repetir ===
mask_postopt_false = results_df['post_optimize'] == False
mask_channels_8_9_nobs_1000 = (results_df['channels'].isin([8, 9])) & (results_df['n_obs'] == 1000)

to_repeat = results_df[mask_postopt_false | mask_channels_8_9_nobs_1000]
print("Configurations to repeat:")
print(to_repeat[['channels', 'n_back', 'max_iter', 'post_optimize', 'n_obs']].drop_duplicates())

# === Nueva lista de resultados ===
new_results = []

for _, row in to_repeat.iterrows():
    n_obs = row['n_obs']
    channels = row['channels']
    n_back = row['n_back']
    max_iter = row['max_iter']
    post_optimize = row['post_optimize']

    # Prepara datos
    if n_obs <= 500:
        df_obs = df_base.iloc[:, :n_obs]
    else:
        original_time = np.linspace(0, 1, 500)
        new_time = np.linspace(0, 1, n_obs)
        interpolated = []
        for idx in range(df_base.shape[0]):
            f = interp1d(original_time, df_base.iloc[idx, :], kind='cubic')
            interpolated.append(f(new_time))
        df_obs = pd.DataFrame(interpolated)

    data_subset = df_obs.iloc[0:channels, :]

    print(f"Re-running: channels={channels}, n_back={n_back}, max_iter={max_iter}, post_optimize={post_optimize}, n_obs={n_obs}")

    times = []
    for repeat in range(N_REPEATS):
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

    new_results.append({
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

# === Guardar resultados corregidos ===
new_results_df = pd.DataFrame(new_results)
print("\nNew Timing summary:")
print(new_results_df)

new_results_df.to_csv('timing_results_corrected.csv', index=False)


#%%
# === Filtro ===
df_plot = results_df[
    (results_df['n_back'] == 5) &
    (results_df['max_iter'] == 10) &
    (results_df['post_optimize'] == True)
]

# Agrupa
grouped = df_plot.groupby(['channels', 'n_obs']).agg(
    time_mean=('time_mean', 'mean'),
    time_min=('time_min', 'min'),
    time_max=('time_max', 'max')
).reset_index()

# Colores de la paleta de Seaborn
palette = sns.color_palette("tab10", n_colors=len(grouped['n_obs'].unique()))
obs_list = sorted(grouped['n_obs'].unique())

plt.figure(figsize=(8, 5))

# Pinta cada grupo con línea + banda
for color, n_obs in zip(palette, obs_list):
    group = grouped[grouped['n_obs'] == n_obs]

    # Línea y puntos sólidos
    plt.plot(
        group['channels'],
        group['time_mean'],
        marker='o',
        label=f'{n_obs}',
        color=color
    )

    # Banda degradada del mismo color con alta transparencia
    plt.fill_between(
        group['channels'],
        group['time_min'],
        group['time_max'],
        color=color,
        alpha=0.2  # Transparencia baja para no tapar la línea
    )

plt.xlabel('Number of channels')
plt.ylabel('Execution time (s)')
plt.legend(title='Number of observations')
plt.grid(True)
plt.savefig('times1.pdf', dpi=300, transparent=True, bbox_inches='tight')

plt.show()


#%%

df_plot2 = results_df[
    (results_df['channels'] == 4) &
    (results_df['n_obs'] == 500)
]

# Agrupa y calcula min–max
grouped = df_plot2.groupby(['max_iter', 'n_back', 'post_optimize']).agg(
    time_mean=('time_mean', 'mean'),
    time_min=('time_min', 'min'),
    time_max=('time_max', 'max')
).reset_index()

# Crea variable combinada para grupos
grouped['group'] = grouped.apply(
    lambda row: f"n_back={row['n_back']}, post_opt={row['post_optimize']}",
    axis=1
)

# Paleta de colores automática
palette = sns.color_palette("tab10", n_colors=grouped['group'].nunique())

plt.figure(figsize=(8, 5))

for color, group_name in zip(palette, grouped['group'].unique()):
    group = grouped[grouped['group'] == group_name]
    plt.plot(
        group['max_iter'],
        group['time_mean'],
        marker='o',
        label=group_name,
        color=color
    )
    plt.fill_between(
        group['max_iter'],
        group['time_min'],
        group['time_max'],
        color=color,
        alpha=0.2
    )

plt.xlabel('Number of iterations (max_iter)')
plt.ylabel('Execution time (s)')
plt.legend(title='Configuration')
plt.grid(True)

# Guarda en PDF y PNG HD
plt.savefig('times2.pdf', bbox_inches='tight')

plt.show()