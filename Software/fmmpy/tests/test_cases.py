# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from fmmpy import fit_fmm

# === Paths to test data ===
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

ECG_PATH = os.path.join(DATA_DIR, "ECG_data.csv")
XPS_PATH = os.path.join(DATA_DIR, "XPS_Fe2p_data.csv")


def test_ecg_fmm_fit():
    df = pd.read_csv(ECG_PATH, header=None).iloc[:, 400:800]

    P_alpha = (4.2, 5.4)
    P_ome = (0.05, 0.25)
    QRS_alpha = (5.4, 6.2)
    QRS_ome = (0.01, 0.10)
    T_alpha = (0, 3.14)
    T_ome = (0.1, 0.5)

    alpha_restr = np.array([P_alpha, QRS_alpha, QRS_alpha, QRS_alpha, T_alpha])
    omega_restr = np.array([P_ome, QRS_ome, QRS_ome, QRS_ome, T_ome])

    model = fit_fmm(
        data_matrix=df.values,
        n_back=5,
        max_iter=3,
        post_optimize=False,  # faster test
        alpha_restrictions=alpha_restr,
        omega_restrictions=omega_restr,
        omega_min=0.01,
        omega_max=0.5
    )

    assert model.n_back == 5
    assert model.n_ch == df.shape[0]
    assert model.prediction.shape == df.shape
    assert np.all(model.R2 >= 0) and np.all(model.R2 <= 1)


def test_xps_fmm_fit():
    df = pd.read_csv(XPS_PATH)
    spectrum = df.iloc[:, 1:5].T.iloc[0, :].values  # use one signal

    model = fit_fmm(
        data_matrix=spectrum,
        n_back=6,
        max_iter=3,
        post_optimize=False,
        omega_min=0.01,
        omega_max=0.2
    )

    assert model.n_back == 6
    assert model.n_ch == 1
    assert model.prediction.shape == (1, spectrum.shape[0])
    assert np.all(model.R2 >= 0) and np.all(model.R2 <= 1)