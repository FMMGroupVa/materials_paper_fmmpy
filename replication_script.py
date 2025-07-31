
import os
from pip._internal.cli.main import main as pip_main

def install_requirements(requirements_path='requirements.txt'):
    if not os.path.isfile(requirements_path):
        raise FileNotFoundError(f"Could not find {requirements_path}")
    
    print(f"Installing from '{requirements_path}' using pip...")
    pip_main(['install', '-r', requirements_path])

def import_dependencies():
    global np, pd, plt, sns, Line2D, AFDCal, interp1d, time, sc, fit_fmm
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D
    from AFDCal import AFDCal
    from scipy.interpolate import interp1d
    import time
    import scipy.signal as sc
    from fmmpy import fit_fmm

def run_script(filename, *args):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            code = file.read()
            print(f"\n--- Running {filename} ---\n")
            exec(code, {'__name__': '__main__', 'args': args})
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"Error executing '{filename}': {e}")

def show_menu():
    print("\n===== SCRIPT MENU =====")
    print("\nReproducing Results")
    print("  1. RUN USE CASES \n\tEstimated running time: ~2 min.")
    print("\nRunning Benchmarks")
    print("  2. Scalability tests (Fig. 4) \n\tEstimated running time: ~12hours (100 REPS)")
    print("  3. AFD-FMM comparison \n\tEstimated running time: ~1 min. (100 REPS)")
    print("  4. Measure impact of restrictions on nonlinear params \n\tEstimated running time: ~1 min. (100 REPS)")
    print("  5. Measure impact of restrictions on linear params (Scalability tests) (Fig. 5)\n\tEstimated running time: ~12hours (100 REPS)")
    print("\n6. Run full script >24h (100 REPS).")
    print("\n0. Exit")

def test_1():
    print("\n--Running: 1. Run Use cases.")

    df = pd.read_csv('Data/ECG_data.csv', header=None)
    df = df.iloc[:,400:800]
    
    # Fit FMM to data
    P_alpha = (4.2, 5.4)
    P_ome = (0.05,0.25)
    QRS_alpha = (5.4, 6.2)
    QRS_ome = (0.01, 0.10)
    T_alpha = (0, 3.14)
    T_ome = (0.1, 0.5)
    
    # Param restriction arguments:
    alpha_restr = np.array([P_alpha, QRS_alpha, QRS_alpha, QRS_alpha, T_alpha])
    omega_restr = np.array([P_ome, QRS_ome, QRS_ome, QRS_ome, T_ome])
    
    res = fit_fmm(data_matrix=df, # Data
                  n_back=5, max_iter=8, post_optimize=True,  # Fit options
                  alpha_restrictions=alpha_restr, omega_restrictions=omega_restr,
                  omega_min=0.01, omega_max=0.5) # Parameter control
    
    # Print model summary:
    print(res)
        
    # Plot results fit:
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    res.plot_predictions(channel_names=lead_names, n_cols=3,
                         width=5.9, height=5, dpi=300, save_path="Results/Figures/ECGfit.png",
                         show=False)
    
    # Plot results residuals:
    res.plot_residuals(channel_names=lead_names, n_cols=4,
                       width=5.9, height=3.5, dpi=300, save_path="Results/Figures/ECGresiduals.png",
                       show=False)
    
    # CIs:
    alpha_ci, omega_ci, delta_ci, gamma_ci = res.conf_intervals(0.95, method=2)
    print(alpha_ci)
    res.show_conf_intervals()
    
    ############################## CASE 2 ##############################
    
    df_raw = pd.read_csv('Data/XPS_Fe2p_data.csv')
    x = df_raw.iloc[:, 0]  
    df = df_raw.iloc[:, 1:5].T
    
    plt.figure(figsize=(6, 4))
    plt.plot(x, df.iloc[0, :])
    plt.xlabel('Binding energy (eV)')
    plt.ylabel('Intensity (u.a.)')
    plt.tight_layout()
    plt.grid()
    plt.savefig("Results/Figures/Fe2p.png", dpi=600, bbox_inches='tight')
    
    spectrum = df.iloc[0, :]
    
    n_back=7
    max_iter=15
    
    res2 = fit_fmm(data_matrix=spectrum, # Data
                   n_back=n_back, max_iter=max_iter, post_optimize=True,  # Fit options
                   omega_min=0.01) # Parameter control
    
    res2.plot_predictions(channels=[0], channel_names=[""],
                          dpi=300, width=2.8, height=2,
                          save_path="Results/Figures/Fe2p_example.png",
                          show=False)
    res2.plot_components(channels=[0], channel_names=[""],
                         dpi=300, width=2.8, height=2,
                         save_path="Results/Figures/Fe2p_comp_example.png",
                         show=False)
    
    alpha_restr = np.array([(0.1,2.9) for i in range(n_back)])
    
    res3 = fit_fmm(data_matrix=spectrum, # Data
                   n_back=n_back, max_iter=max_iter, post_optimize=True,  # Fit options
                   omega_min=0.01, omega_max=0.1,
                   length_alpha_grid=100, alpha_restrictions=alpha_restr,
                   beta_min=np.pi-0.25, beta_max=np.pi+0.25)
    
    res3.plot_predictions(channels=[0], channel_names=[""],
                         dpi=300, width=2.8, height=2,
                         save_path="Results/Figures/Fe2p_example_restr.png",
                         show=False)
    res3.plot_components(channels=[0], channel_names=[""],
                         dpi=300, width=2.8, height=2,
                         save_path="Results/Figures/Fe2p_comp_example_restr.png",
                         show=False)

def test_2(N_REPEATS):
    print("\nRunning: 2. Scalability tests (Fig. 4).")
    
    # ====== Data: ECG beat (Could be simulated) ======
    df = pd.read_csv('Data/ECG_data.csv', header=None)
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
            print(f"Run: n_obs={n_obs}, channels={channels}")
            data_subset = df_obs.iloc[0:channels, :]
            for n_back in n_back_values:
                
                for max_iter in max_iter_values:
                    for post_optimize in post_optimize_values:
                        times = []
                        for repeat in range(N_REPEATS):
                            start = time.perf_counter()
                            _ = fit_fmm(
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
    results_df.to_csv('Results/timing_results_repeated_nobs.csv', index=False)
    
    # ========= Plot Generation =========
    results_df = pd.read_csv('Results/timing_results_repeated_nobs.csv')
    
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

def test_3(N_REPEATS):
    print("\n--Running: 3. AFD-FMM comparison.")
        
    df = pd.read_csv('Data/ECG_data.csv', header=None)
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
        _ = fit_fmm(
            data_matrix=df_base,
            n_back=5,
            max_iter=1,
            post_optimize=False,
            length_omega_grid=50,
            omega_min=0.01,
            omega_max=0.5
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
    
def test_4(N_REPEATS):
    print("\n--Running: 4. Restrictions on nonlinear params.")
    
    df = pd.read_csv('Data/ECG_data.csv', header=None)
    df_base = df.iloc[:, 350:850]  # 500 obs
    
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
        _ = fit_fmm(
            data_matrix=df_base,
            n_back=5,
            max_iter=8,
            post_optimize=True,
            alpha_restrictions=alpha_restr,
            omega_restrictions=omega_restr,
            omega_min=0.01,
            omega_max=0.5)
        end = time.perf_counter()
        times_restricted.append(end - start)
    
    # === Fitting sin restricciones ===
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        _ = fit_fmm(
            data_matrix=df_base,
            n_back=5,
            max_iter=8,
            post_optimize=True,
            omega_min=0.01,
            omega_max=0.5
        )
        end = time.perf_counter()
        times_unrestricted.append(end - start)
    
    # === Estadísticos ===
    mean_restricted = np.mean(times_restricted)
    max_restricted = np.max(times_restricted)
    mean_unrestricted = np.mean(times_unrestricted)
    max_unrestricted = np.max(times_unrestricted)
    
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

def test_5(N_REPEATS):
    print("\n--Running: 5. Restrictions on linear params.")
    
    df_raw = pd.read_csv('Data/XPS_Fe2p_data.csv')
    df_signal = df_raw.iloc[:, 1:5].T
    
    # === Config general ===
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
            print(f'Run: n_obs={n_obs}, channels={channels}')
            # Repetir si hace falta
            times_to_repeat = int(np.ceil(channels / df_obs.shape[0]))
            df_expanded = pd.concat([df_obs] * times_to_repeat, ignore_index=True)
            df_subset = df_expanded.iloc[0:channels, :]

            times = []
            for _ in range(N_REPEATS):
                start = time.perf_counter()
                _ = fit_fmm(
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
                    _ = fit_fmm(
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
    
def main():
    install_requirements('requirements.txt')
    import_dependencies()
    while True:
        show_menu()
        try:
            choice = input("\nEnter your choice (1-6, or 0 to exit): ").strip()
            if choice == '0':
                print("Exiting.")
                break
            
            if choice != '1':
                nreps = input("\nEnter the number of repetitions for each test (100 by default): ").strip()

                if nreps.isdigit():
                    nreps = int(nreps)
                else:
                    nreps = 100
            
            if choice == '1':
                test_1()
                
            elif choice == '2':
                test_2(nreps)
                
            elif choice == '3':
                test_3(nreps)
                
            elif choice == '4':
                test_4(nreps)
                
            elif choice == '5':
                test_5(nreps)
                
            elif choice == '6':
                test_1()
                test_2(nreps)
                test_3(nreps)
                test_4(nreps)
                test_5(nreps)
                
            else:
                print("Invalid selection. Please choose a number from the menu.")
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
