"""Functions for calculating ensemble averages"""

import numpy as np
from operator import itemgetter
import os.path

from pymbar import timeseries

from origamipy.op_process import *


def calc_pmf(weights):
    """Convert weights to potential of mean force

    Uses largest weight as reference.
    """
    max_weight = max(weights)
    pmfs = []
    for weight in weights:
        if weight != 0:
            pmfs.append(np.log(max_weight / weight))
        else:
            pmfs.append('nan')

    return pmfs


def calc_pmf_with_stds(weights, weight_stds):
    """Convert weights to potential of mean force

    Uses largest weight as reference.
    """
    max_weight = max(weights)
    max_weight_i = weights.index(max_weight)
    max_weight_std = weight_stds[max_weight_i]
    pmfs = []
    pmf_stds = []
    for weight, std in zip(weights, weight_stds):
        if weight != 0:
            pmfs.append(np.log(max_weight / weight))
            e1 = max_weight_std / max_weight
            e2 = std / weight
            e = np.sqrt(e1**2 + e2**2)
            pmf_stds.append(e)
        else:
            pmfs.append('nan')
            pmf_stds.append('nan')

    return pmfs, pmf_stds
