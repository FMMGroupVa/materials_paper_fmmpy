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
    """
    Represents a fitted Frequency Modulated Möbius (FMM) model.

    This class encapsulates the result of applying FMM decomposition to multichannel
    signals. It stores the original data, estimated parameters, predicted signal,
    and performance metrics such as R² and partial R². It also provides utilities 
    for visualizing fitted components, residuals, and confidence intervals.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_timepoints)
        Original input signal(s) used in the fitting process.
    
    time_points : array-like
        Time points corresponding to the signal observations.
    
    prediction : np.ndarray
        Predicted signal obtained from the fitted model.
    
    params : dict
        Dictionary with fitted parameters. Must include at least 'alpha'.
        Other typical entries are: 'omega', 'A', 'beta', 'delta', 'gamma', 'phi', and 'coef'.
    
    restricted : bool, optional
        Whether the fitting was done with constraints (default is False).
    
    max_iter : int, optional
        Maximum number of iterations used in the fitting algorithm.

    Attributes
    ----------
    data : np.ndarray
        Original input data.
    
    time_points : np.ndarray
        Time points of the signal.
    
    prediction : np.ndarray
        Model prediction (real part).
    
    params : dict
        Ordered fitted parameters for the model.
    
    restricted : bool
        Indicates if the model was fitted with parameter restrictions.
    
    max_iter : int
        Number of iterations used in model fitting.
    
    n_ch : int
        Number of channels in the signal.
    
    n_obs : int
        Number of observations per channel.
    
    n_back : int
        Number of FMM components used in the model.
    
    R2 : np.ndarray
        R² values for each channel.
    
    partial_R2 : np.ndarray
        Partial R² values for each component and channel.
    
    sigma : np.ndarray
        Standard deviation of the residuals per channel.

    Methods
    -------
    show()
        Prints a summary of the fitted model.
    
    predict(X)
        [Not implemented] Placeholder for prediction at arbitrary time points.
    
    plot_predictions(...)
        Plot the fitted signal and the observed signal for selected channels.
    
    plot_residuals(...)
        Plot the residuals (difference between observed and predicted).
    
    plot_components(...)
        Plot the contribution of each FMM component per channel.
    
    calculate_SE(method=2)
        Compute standard errors of the estimated parameters using a linearized model.
    
    conf_intervals(conf_level=0.95, method=2)
        Compute confidence intervals for key parameters.
    
    show_conf_intervals(conf_level=0.95, method=2)
        Print formatted confidence intervals.
    
    calculate_partial_R2()
        Compute partial R² values for each component in each channel.
    """
    def __init__(self, data=None, time_points=None,  prediction=None, params=None, restricted=False, max_iter=None):
        """
        Initialize an instance of the FMMModel class.
        
        This constructor processes and stores the result of fitting an FMM model,
        including the original data, prediction, estimated parameters, and 
        evaluation metrics such as R² and partial R².
        
        Parameters
        ----------
        data : np.ndarray
            Matrix of original input signals with shape (n_channels, n_timepoints).
        
        time_points : array-like
            Vector of time points corresponding to the signal samples.
        
        prediction : np.ndarray
            Fitted signal obtained from the FMM decomposition.
        
        params : dict
            Dictionary containing estimated parameters. Must include at least 'alpha'.
            Each value can be a 1D or 2D array and may include:
            - 'alpha', 'omega', 'A', 'beta', 'delta', 'gamma' (core parameters),
            - 'phi' and 'coef' (complex and real-valued representations),
            - 'M' (mean level).
        
        restricted : bool, optional
            Whether the model was fitted with parameter restrictions. Default is False.
        
        max_iter : int, optional
            Maximum number of iterations used in the fitting procedure.
        
        Raises
        ------
        ValueError
            If `params` is None or does not include the 'alpha' key.
            If the shape of a parameter array is not compatible with the number of components.
            If an unsupported number of dimensions is found in a parameter array.
        """
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
        """
        Print a human-readable summary of the fitted FMM model.
        
        Displays key information about the model, including:
        - Number of channels and components
        - Number of iterations used
        - Estimated alpha and omega parameters
        - Mean partial R² per component
        - R² for each channel
        
        The output is printed directly to the standard output.
        """
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
        """
        Return a formatted string representation of the fitted FMM model.
        
        The output includes:
        - Model type (restricted or unrestricted)
        - Number of channels, components, and iterations
        - Estimated alphas and omegas
        - Mean partial R²
        - R² for each channel (in one or two lines)
        
        Returns
        -------
        str
            A multi-line string summarizing the model.
        """
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
        """
        Plot the original signal and the fitted FMM prediction for selected channels.
    
        This method generates one subplot per selected channel, showing both the 
        observed signal and the predicted signal over time. Axis ticks are adapted 
        for periodic data in [0, 2π].
    
        Parameters
        ----------
        channels : list of int or None, optional
            Indices of channels to plot. If None, all channels are plotted.
    
        channel_names : list of str or None, optional
            Custom names for each channel to be used in plot titles. If None, 
            channels are labeled numerically.
    
        n_cols : int or None, optional
            Number of columns in the subplot grid. If None, it is automatically determined.
    
        save_path : str or None, optional
            If provided, the figure is saved to this path (in PNG format).
            If None, the figure is displayed on screen.
    
        height : float or None, optional
            Height of the figure in inches. If None, it is scaled based on the number of rows.
    
        width : float or None, optional
            Width of the figure in inches. If None, it is scaled based on the number of columns.
    
        dpi : int or None, optional
            Dots per inch for the figure resolution.
    
        Returns
        -------
        None
        """
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
        """
        Plot the residuals (difference between original and predicted signal) for selected channels.
        
        This method generates one subplot per selected channel, displaying the 
        residual signal (data - prediction) over time. A horizontal dashed line 
        at zero is added for reference. X-axis ticks are adapted for periodic signals in [0, 2π].
        
        Parameters
        ----------
        channels : list of int or None, optional
            Indices of channels to plot. If None, all channels are plotted.
        
        channel_names : list of str or None, optional
            Custom names for each channel to be used in plot titles. If None, 
            channels are labeled numerically.
        
        n_cols : int or None, optional
            Number of columns in the subplot grid. If None, it is automatically determined.
        
        save_path : str or None, optional
            If provided, the figure is saved to this path (in PNG format).
            If None, the figure is displayed on screen.
        
        height : float or None, optional
            Height of the figure in inches. If None, it is scaled based on the number of rows.
        
        width : float or None, optional
            Width of the figure in inches. If None, it is scaled based on the number of columns.
        
        dpi : int or None, optional
            Dots per inch for the figure resolution.
        
        Returns
        -------
        None
        """
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
        """
        Plot the contribution of each FMM component for selected channels.
    
        This method shows each FMM component individually as a waveform centered at zero,
        allowing visual inspection of their shape and asymmetry. One subplot is generated 
        per selected channel.
    
        Parameters
        ----------
        n_obs : int or None, optional
            Number of time points to use for plotting. If None, the full length of the signal is used.
    
        channels : list of int or None, optional
            Indices of channels to plot. If None, all channels are plotted.
    
        channel_names : list of str or None, optional
            Custom names for each channel to be used in plot titles. If None, 
            channels are labeled numerically.
    
        n_cols : int or None, optional
            Number of columns in the subplot grid. If None, it is automatically determined.
    
        save_path : str or None, optional
            If provided, the figure is saved to this path (in PNG format).
            If None, the figure is displayed on screen.
    
        height : float or None, optional
            Height of the figure in inches. If None, it is scaled based on the number of rows.
    
        width : float or None, optional
            Width of the figure in inches. If None, it is scaled based on the number of columns.
    
        dpi : int or None, optional
            Dots per inch for the figure resolution.
    
        Returns
        -------
        None
        """
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
        
        waves = self._get_waves_ch(n_obs)
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
        """
        Compute the standard errors of the FMM parameters using a linearized approximation.
        
        This method builds a local linear model around the fitted signal and computes
        standard errors for the parameters M, alpha, omega, delta, and gamma using
        either homoscedastic or heteroscedastic assumptions.
        
        Parameters
        ----------
        method : int, optional
            Method used for variance estimation:
            - 1: Homoscedastic assumption (common variance across channels)
            - 2: Heteroscedastic (channel-specific variances) [default]
            - 3: Sandwich estimator (heteroscedastic, no structure assumed)
        
        Returns
        -------
        dict
            A dictionary with entries:
            - 'M'     : array of shape (n_channels,)
            - 'alpha' : array of shape (n_back,)
            - 'omega' : array of shape (n_back,)
            - 'delta' : array of shape (n_channels, n_back)
            - 'gamma' : array of shape (n_channels, n_back)
            containing the standard errors for each parameter.
        """
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
        """
        Compute confidence intervals for the key FMM parameters.
        
        Confidence intervals are derived assuming asymptotic normality of the estimators
        and using standard errors computed via `calculate_SE`.
        
        Parameters
        ----------
        conf_level : float, optional
            Confidence level for the intervals (e.g., 0.95 for 95%). Default is 0.95.
        
        method : int, optional
            Method to compute standard errors (same as in `calculate_SE`).
        
        Returns
        -------
        tuple
            A tuple of four confidence intervals:
            - alpha_ci : (lower_bounds, upper_bounds), each of shape (n_back,)
            - omega_ci : (lower_bounds, upper_bounds), each of shape (n_back,)
            - delta_ci : (lower_bounds, upper_bounds), each of shape (n_channels, n_back)
            - gamma_ci : (lower_bounds, upper_bounds), each of shape (n_channels, n_back)
        """
        SE = self.calculate_SE(method=method)
        z = norm.ppf(0.5 + conf_level/ 2)
        alpha_ci = ((self.params['alpha']-z*SE['alpha']) % (2*np.pi), (self.params['alpha']+z*SE['alpha']) % (2*np.pi))
        omega_ci = (self.params['omega']-z*SE['omega'], self.params['omega']+z*SE['omega'])
        delta_ci = (self.params['delta']-z*SE['delta'], self.params['delta']+z*SE['delta'])
        gamma_ci = (self.params['gamma']-z*SE['gamma'], self.params['gamma']+z*SE['gamma'])
        return alpha_ci, omega_ci, delta_ci, gamma_ci
        
    def show_conf_intervals(self, conf_level=0.95, method=2):
        """
        Print formatted confidence intervals for alpha and omega parameters.
        
        Internally calls `conf_intervals` and displays the intervals in tabular form
        for each FMM component.
        
        Parameters
        ----------
        conf_level : float, optional
            Confidence level for the intervals (e.g., 0.95 for 95%). Default is 0.95.
        
        method : int, optional
            Method to compute standard errors (same as in `calculate_SE`).
        
        Returns
        -------
        None
        """
        alpha_ci, omega_ci, delta_ci, gamma_ci = self.conf_intervals(conf_level, method=method)
    
        print(f"Confidence Intervals ({int(conf_level*100)}%)")
        print("-" * 32)
        print(f"{'Component':<10} | {'α lower':>8}  {'α upper':>8} | {'ω lower':>8}  {'ω upper':>8}")
        for k in range(self.n_back):
            print(f"{k+1:<10} | {alpha_ci[0][k]:>8.3f}  {alpha_ci[1][k]:>8.3f} | "
                  f"{omega_ci[0][k]:>8.3f}  {omega_ci[1][k]:>8.3f}")
    
    def _get_waves_ch(self, n_obs):
        """
        Return the individual FMM components for each channel as real-valued waveforms.
        
        Each component is obtained by applying the corresponding Möbius function to its 
        estimated parameter `a_k` and scaling it by its complex coefficient `phi_k`.
        
        Parameters
        ----------
        n_obs : int
            Number of time points to use for waveform generation.
        
        Returns
        -------
        list of np.ndarray
            A list with length equal to the number of channels.
            Each element is a 2D array of shape (n_back, n_obs) containing the real-valued 
            contributions of each component for that channel.
        """
        waves = [np.zeros((self.n_back, n_obs))] * self.n_ch
        t = seq_times(n_obs)
        for ch_i in range(self.n_ch):
            for k in range(self.n_back):
                waves[ch_i][k,:] = (self.params['phi'][ch_i, k+1]*mobius(self.params['a'][k+1], t)).real
        return waves
    
    def calculate_partial_R2(self):
        """
        Compute the partial R² values for each FMM component and channel.
    
        For each component \( k \), the method computes the increase in explained
        variance when including component \( k \) in the model. It does so by comparing
        the residual sum of squares (RSS) of the full model with that of a reduced model
        excluding component \( k \).
    
        The result helps quantify the relative importance of each component in explaining
        the signal in each channel.
    
        Returns
        -------
        np.ndarray
            A matrix of shape (n_channels, n_back) where each entry [i, k] represents
            the partial R² of component k in channel i.
        """
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
    
    
    
    
    
