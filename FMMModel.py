# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 10:18:05 2025

@author: Christian
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.linalg import block_diag
from scipy.stats import norm
from auxiliar_functions import mobius, seq_times

class FMMModel:
    def __init__(self, data=None, time_points=None,  prediction=None, params=None, restricted=False, max_iter=None):
        
        self.data = data
        self.time_points = time_points
        self.prediction = prediction.real
        self.max_iter = max_iter
        
        if params is None or 'alpha' not in params:
            raise ValueError("Parameter dictionary must include 'alpha'.")

        alpha = np.array(params['alpha'])
        K = len(alpha)  # number of components
        order = np.argsort((alpha + np.pi) % (2 * np.pi))

        self.params = {}
        
        do_not_order_keys = ['M'] 
        for key, value in params.items():
            if key in do_not_order_keys:
                self.params[key] = np.array(value)
                continue
        
            arr = np.array(value)
        
            if arr.ndim == 1:
                if arr.shape[0] == K + 1:
                    fixed = arr[0:1]
                    reordered = arr[1:][order]
                    self.params[key] = np.concatenate([fixed, reordered])
                elif arr.shape[0] == K:
                    self.params[key] = arr[order]
                else:
                    raise ValueError(f"Unexpected shape for param '{key}': {arr.shape}")
        
            elif arr.ndim == 2:
                if arr.shape[1] == K + 1:
                    fixed = arr[:, [0]]
                    reordered = arr[:, 1:][:, order]
                    self.params[key] = np.hstack([fixed, reordered])
                elif arr.shape[1] == K:
                    self.params[key] = arr[:, order]
                else:
                    raise ValueError(f"Unexpected shape for param '{key}': {arr.shape}")
        
            else:
                raise ValueError(f"Unsupported number of dimensions for param '{key}': {arr.ndim}")
        
        self.n_ch, self.n_obs = data.shape
        self.n_back = len(params['alpha'])
        var_data = np.var(data, axis=1)
        var_error = np.var(data-prediction, axis=1)
        
        self.sigma = np.sqrt(var_error)
        self.R2 = 1-var_error/var_data
        self.partial_R2 = self.calculate_partial_R2()
        self.restricted = restricted
        
    #POR DEFINIR PARA t ARBITRARIO
    def predict(self, X):
        
        return 0
    
    def show(self):
        header = "Restricted FMM Model" if self.restricted else "FMM Model"
        print(header)
        print("-" * len(header))
        
        print(f"{'Channels':<15}: {self.n_ch}")
        print(f"{'Components':<15}: {self.n_back}")
        print(f"{'Max. iterations':<15}: {self.max_iter}")
        
        print(f"{'Alphas':<15}: " + "  ".join(f"{a:.3f}" for a in self.params['alpha']))
        print(f"{'Omegas':<15}: " + "  ".join(f"{o:.3f}" for o in self.params['omega']))
        print(f"{'Mean partial R²':<15}: " + "  ".join(f"{r:.3f}" for r in np.mean(self.partial_R2, axis=0)))
        
        r2_values = [f"{r:.3f}" for r in self.R2]
        halfway = len(r2_values) // 2 + len(r2_values) % 2
        print(f"{'R² per channel':<15}: " + "  ".join(r2_values[:halfway]))
        if len(r2_values) > halfway:
            print(" " * 17 + "  ".join(r2_values[halfway:]))
    
    def __str__(self):
        header = "Restricted FMM Model" if self.restricted else "FMM Model"
        lines = [
            header,
            "-" * len(header),
            f"{'Channels':<15}: {self.n_ch}",
            f"{'Components':<15}: {self.n_back}",
            f"{'Max. iterations':<15}: {self.max_iter}",
            f"{'Alphas':<15}: " + "  ".join(f"{a:.3f}" for a in self.params['alpha']),
            f"{'Omegas':<15}: " + "  ".join(f"{o:.3f}" for o in self.params['omega']),
            f"{'Mean partial R²':<15}: " + "  ".join(f"{r:.3f}" for r in np.mean(self.partial_R2, axis=0))
        ]
    
        # R² per channel in two lines if needed
        r2_values = [f"{r:.3f}" for r in self.R2]
        halfway = len(r2_values) // 2 + len(r2_values) % 2
        r2_line1 = f"{'R² per channel':<15}: " + "  ".join(r2_values[:halfway])
        r2_line2 = " " * 17 + "  ".join(r2_values[halfway:]) if len(r2_values) > halfway else ""
    
        lines.append(r2_line1)
        if r2_line2:
            lines.append(r2_line2)
    
        return "\n".join(lines)
    
    def plot_predictions(self, channels=None, channel_names=None, n_cols=None,
                         save_path=None, height=None, width=None, dpi=None):
        
        # Channel selection
        if channels is None:
            channels = np.arange(self.n_ch)
            
        if len(channels) > self.n_ch:
            channels = channels[0:self.n_ch]
            
        n = len(channels)
        
        # Layout definition
        if n_cols is None:
            n_cols = math.ceil(math.sqrt(n))
        n_cols = min(n_cols, n)
        n_rows = math.ceil(n / n_cols)
        
        # Fig dimensions (in inches)
        if height is None:
            height = 3 * n_rows
        if width is None:
            width = 4 * n_cols
        
        # Autoscale font size based on figure area
        fig_area = width * height
        base_fontsize = max(6, min(12, fig_area * 0.1))  # between 6 and 12 pts
        
        # Figure definition
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), 
                                 squeeze=False, dpi=dpi, constrained_layout=True)

        if not isinstance(channel_names, list) and channel_names is not None:
            channel_names = [channel_names]
    
        for idx, ch in enumerate(channels):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row][col]
    
            if 0 <= ch < self.n_ch:
                ax.plot(self.time_points[0], self.data[ch, :], label="Data", color="tab:blue")
                ax.plot(self.time_points[0], self.prediction[ch, :].real, label="Prediction", color="tab:orange")
                
                # Set title
                if channel_names is not None and ch < len(channel_names): 
                    name = channel_names[ch]
                    ax.set_title(f"{name}", fontsize=base_fontsize + 2)
                else:
                    ax.set_title(f"Channel {ch}", fontsize=base_fontsize + 2)
                
                # Grid and ticks
                ax.grid(True)
                ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
                ax.set_xticklabels([]) 
                ax.tick_params(axis='x', which='both', length=0)
                ax.tick_params(axis='y', labelsize=base_fontsize)
            else:
                ax.set_title(f"Channel {ch} (out of range)", fontsize=base_fontsize)
                ax.axis("off")
        
        # Turn off unused subplots
        for idx in range(n, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].axis("off")
    
        # Save or show
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_residuals(self, channels=None, channel_names=None, n_cols=None,
                   save_path=None, height=None, width=None, dpi=None):
    
        # Channel selection
        if channels is None:
            channels = np.arange(self.n_ch)
            
        if len(channels) > self.n_ch:
            channels = channels[0:self.n_ch]
            
        n = len(channels)
        
        # Layout definition
        if n_cols is None:
            n_cols = math.ceil(math.sqrt(n))
        n_cols = min(n_cols, n)
        n_rows = math.ceil(n / n_cols)
        
        # Fig dimensions (in inches)
        if height is None:
            height = 3 * n_rows
        if width is None:
            width = 4 * n_cols
        
        # Autoscale font size based on figure area
        fig_area = width * height
        base_fontsize = max(6, min(12, fig_area * 0.1))  # between 6 and 12 pts
        
        # Figure definition
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), 
                                 squeeze=False, dpi=dpi, constrained_layout=True)
    
        if not isinstance(channel_names, list) and channel_names is not None:
            channel_names = [channel_names]
    
        for idx, ch in enumerate(channels):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row][col]
    
            if 0 <= ch < self.n_ch:
                residual = self.data[ch, :] - self.prediction[ch, :].real
                ax.plot(self.time_points[0], residual, color="tab:red", label="Residual")
                ax.axhline(0, linestyle='--', color='black', linewidth=0.8)
                
                # Title
                if channel_names is not None and ch < len(channel_names): 
                    name = channel_names[ch]
                    ax.set_title(f"{name}", fontsize=base_fontsize + 2)
                else:
                    ax.set_title(f"Channel {ch}", fontsize=base_fontsize + 2)
                
                # Grid and ticks
                ax.grid(True)
                ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
                ax.set_xticklabels([]) 
                ax.tick_params(axis='x', which='both', length=0)
                ax.tick_params(axis='y', labelsize=base_fontsize)
            else:
                ax.set_title(f"Channel {ch} (out of range)", fontsize=base_fontsize)
                ax.axis("off")
        
        # Turn off unused subplots
        for idx in range(n, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].axis("off")
    
        # Save or show
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_components(self, n_obs=None, channels=None, channel_names=None, n_cols=None,
                        save_path=None, height=None, width=None, dpi=None):
    
        # Selección de canales
        if channels is None:
            channels = np.arange(self.n_ch)
        n = len(channels)
        
        if n_obs is None:
            n_obs = self.n_obs
        # Layout
        if n_cols is None:
            n_cols = math.ceil(math.sqrt(n))
        n_cols = min(n_cols, n)
        n_rows = math.ceil(n / n_cols)
        
        # Dimensiones de figura
        if height is None:
            height = 3 * n_rows
        if width is None:
            width = 4 * n_cols
            
        fig_area = width * height
        base_fontsize = max(6, min(12, fig_area * 0.1))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height),
                                 squeeze=False, dpi=dpi, constrained_layout=True)
        
        # Normalización de nombres
        if not isinstance(channel_names, list) and channel_names is not None:
            channel_names = [channel_names]
        
        # Color por componente
        colors = plt.cm.tab10.colors
        
        waves = self.get_waves_ch(n_obs)
        time_points = seq_times(n_obs)[0]
        for idx, ch in enumerate(channels):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row][col]
    
            if 0 <= ch < self.n_ch:
                components = waves[ch]  # shape: (n_components, n_time_points)
                for i, comp in enumerate(components):
                    comp2 = comp - comp[0]
                    ax.plot(time_points, comp2, label=f"Comp {i+1}", color=colors[i % len(colors)])
                
                # Título
                if channel_names is not None and ch < len(channel_names):
                    name = channel_names[ch]
                    ax.set_title(f"{name}", fontsize=base_fontsize + 2)
                else:
                    ax.set_title(f"Channel {ch}", fontsize=base_fontsize + 2)
                
                ax.grid(True)
                ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', length=0)
                ax.tick_params(axis='y', labelsize=base_fontsize)
            else:
                ax.set_title(f"Channel {ch} (out of range)", fontsize=base_fontsize)
                ax.axis("off")
    
        # Apagar subplots sobrantes
        for idx in range(n, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].axis("off")
        
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
        plt.show()
    
    def calculate_SE(self, method=2):

        ts = [2*np.arctan(self.params['omega'][k]*np.tan((self.time_points[0]-self.params['alpha'][k])/2)) for k in range(self.n_back)]
        
        intercepts_block = block_diag(*[np.ones((self.n_obs,1)) for _ in range(self.n_ch)])
        
        # Order by channels: (delta_1(1), ..., delta_K(1), ..., delta_1(L), ..., delta_K(L),)
        delta_block_diag = block_diag(*[np.stack([np.cos(tsk) for tsk in ts], axis=1) for _ in range(self.n_ch)])
        gamma_block_diag = block_diag(*[np.stack([np.sin(tsk) for tsk in ts], axis=1) for _ in range(self.n_ch)])
        
        factor_1_alphas = [(self.params['omega'][k] + (1-self.params['omega'][k]**2)*(1-np.cos(ts[k])/(2*self.params['omega'][k]))) for k in range(self.n_back)] 
        factor_1_omegas = [np.sin(ts[k]) / self.params['omega'][k]  for k in range(self.n_back)]
        alpha_block = [None]*self.n_ch
        omega_block = [None]*self.n_ch
        for ch in range(self.n_ch):
            factor_2 = [(self.params['delta'][ch,k]*np.sin(ts[k]) - self.params['gamma'][ch,k]*np.cos(ts[k])) for k in range(self.n_back)] 
            alpha_block[ch] = np.stack([factor_1_alphas[k] * factor_2[k] for k in range(self.n_back)], axis=1)
            omega_block[ch] = np.stack([factor_1_omegas[k] * -factor_2[k] for k in range(self.n_back)], axis=1)
        
        alpha_block = np.vstack(alpha_block)
        omega_block = np.vstack(omega_block)
        
        F0 = np.hstack([intercepts_block, alpha_block, omega_block, delta_block_diag, gamma_block_diag])
        
        # Version 1 - Homocedastic case
        if method == 1:
            SE_mat = np.linalg.inv(F0.T @ F0)
            common_var = np.mean(self.sigma**2)
            SE_params = np.sqrt(common_var*np.diag(SE_mat))
            
        # Version 2 - Heterocedastic case (Sigma per channel)
        if method == 2:
            W = np.repeat(1/self.sigma, self.n_obs)
            F0 = F0 * W[:, np.newaxis]
            SE_mat = np.linalg.inv(F0.T @ F0)
            SE_params = np.sqrt(np.diag(SE_mat))
        
        # Version 3 - Heterocedastic case (No sigma estructure - Sandwich Estimator)
        if method == 3:
            residuals = self.data.flatten() - self.prediction.flatten()  # tamaño (nL,)
            W = np.diag(residuals**2) 
            # W = np.diag(np.repeat(self.sigma**2, self.n_obs))
            SE_mat = np.linalg.inv(F0.T @ F0) @ F0.T @ W @ F0 @ np.linalg.inv(F0.T @ F0)
            SE_params = np.sqrt(np.diag(SE_mat))
        
        SE = {'M': SE_params[0 : self.n_ch],
              'alpha': SE_params[self.n_ch : self.n_ch+self.n_back],
              'omega': SE_params[self.n_ch+self.n_back : self.n_ch+2*self.n_back],
              # Order by channels -> Reconstruct by rows 
              'delta': SE_params[self.n_ch+2*self.n_back : self.n_ch+2*self.n_back+self.n_ch*self.n_back].reshape(self.n_ch, self.n_back),
              'gamma': SE_params[self.n_ch+2*self.n_back+self.n_ch*self.n_back : self.n_ch+2*self.n_back*(self.n_ch+1)].reshape(self.n_ch, self.n_back)}
        
        return SE
    
    def conf_intervals(self, conf_level=0.95, method=2):
        SE = self.calculate_SE(method=method)
        z = norm.ppf(0.5 + conf_level/ 2)
        alpha_ci = ((self.params['alpha']-z*SE['alpha']) % (2*np.pi), (self.params['alpha']+z*SE['alpha']) % (2*np.pi))
        omega_ci = (self.params['omega']-z*SE['omega'], self.params['omega']+z*SE['omega'])
        delta_ci = (self.params['delta']-z*SE['delta'], self.params['delta']+z*SE['delta'])
        gamma_ci = (self.params['gamma']-z*SE['gamma'], self.params['gamma']+z*SE['gamma'])
        return alpha_ci, omega_ci, delta_ci, gamma_ci
        
    def show_conf_intervals(self, conf_level=0.95, method=2):
        alpha_ci, omega_ci, delta_ci, gamma_ci = self.conf_intervals(conf_level, method=method)
    
        print(f"Confidence Intervals ({int(conf_level*100)}%)")
        print("-" * 32)
        print(f"{'Component':<10} | {'α lower':>8}  {'α upper':>8} | {'ω lower':>8}  {'ω upper':>8}")
        for k in range(self.n_back):
            print(f"{k+1:<10} | {alpha_ci[0][k]:>8.3f}  {alpha_ci[1][k]:>8.3f} | "
                  f"{omega_ci[0][k]:>8.3f}  {omega_ci[1][k]:>8.3f}")
    
    def get_waves_ch(self, n_obs):
        waves = [np.zeros((self.n_back, n_obs))] * self.n_ch
        t = seq_times(n_obs)
        for ch_i in range(self.n_ch):
            for k in range(self.n_back):
                waves[ch_i][k,:] = (self.params['phi'][ch_i, k+1]*mobius(self.params['a'][k+1], t)).real
        return waves
    
    def calculate_partial_R2(self):
        alphas = self.params['alpha']
        omegas = self.params['omega']
        time_points = self.time_points
        ts = [2*np.arctan(omegas[i] * np.tan((time_points[0] - alphas[i])/2)) for i in range(self.n_back)]
        DM = np.column_stack([np.ones(self.n_obs)] + [np.column_stack([np.cos(ts[i]), np.sin(ts[i])]) for i in range(self.n_back)])
        
        partial_R2 = np.zeros((self.n_ch, self.n_back))
        RSE = (self.n_obs-1)*np.var(self.data - self.prediction, axis=1)
        
        for k in range(self.n_back):
            DM_k = np.delete(DM, [2*k + 1, 2*k + 2], axis=1)
            # RLS[ch_i] = solve_qp(DM.T @ DM, -DM.T@data_matrix[ch_i], G=G, h=h, solver='quadprog')
            estim = np.linalg.inv(DM_k.T @ DM_k) @ DM_k.T @ self.data.T
            prediction = np.dot(DM_k, estim)
            squared_errors_k = (self.data - prediction.T)**2
            RSE_k = np.sum(squared_errors_k, axis=1)
            partial_R2[:,k] = (RSE_k-RSE)/RSE_k
        
        return partial_R2
    
    
    
    
    
