# -*- coding: utf-8 -*-

import numpy as np

def seq_times(nObs):
    return np.reshape(np.linspace(0, 2 * np.pi, num=nObs+1)[:-1], (1,nObs))

    