"""Functions for carrying out Multi Bennet Acceptance Ration (MBAR) analysis"""

import math

from pymbar import timeseries
import scipy

from origamipy.us_process import *


def calc_rstaple_u(staple_M, lattice_site_volume):
    """Calculate reduced staple chemical potential"""
    sites_per_litre = 1e-3 / lattice_site_volume
    rstaple_u = math.log(staple_M * scipy.constants.N_A / sites_per_litre)
    return rstaple_u 


def calc_correlated_rpots(wins, win_enes, win_ops, win_biases, rstaple_u, tags,
        min_bias, slope):
    correlated_rpots = []
    for i in range(len(wins)):
        rpots = calc_reduced_potentials(wins[i], win_enes[i], win_ops[i],
                win_biases[i], rstaple_u, tags, min_bias, slope)
        correlated_rpots.append(rpots)

    return correlated_rpots


def calc_reduced_potentials(win, win_enes, win_ops, win_biases, rstaple_u, tags,
        min_bias, slope):
    """Calculate reduced potentials as defined in shirts2008"""
    min_op1 = win[0][0]
    max_op1 = win[1][0]
    min_op2 = win[0][1]
    max_op2 = win[1][1]
    reduced_potentials = []
    for i in range(len(win_enes)):
        num_staples = win_ops['numstaples'][i]
        op1 = win_ops[tags[0]][i]
        op2 = win_ops[tags[1]][i]
        rchem_pot = num_staples * rstaple_u
        bias = 0
        if op1 < min_op1:
            bias += min_bias + slope*(min_op1 - op1)
        if op1 > max_op1:
            bias += min_bias + slope*(op1 - max_op1)
        if op2 < min_op2:
            bias += min_bias + slope*(min_op2 - op2)
        if op2 > max_op2:
            bias += min_bias + slope*(op2 - max_op2)

        point = (int(op1), int(op2))
        if point in win_biases:
            bias += win_biases[point]

        rpot = win_enes[i] + bias + rchem_pot
        reduced_potentials.append(rpot)

    return reduced_potentials


def calc_no_bias_reduced_potentials(enes, ops, rstaple_u):
    """Calculate reduced potentials as defined in shirts2008"""
    reduced_potentials = []
    for i in range(len(enes)):
        num_staples = ops['numstaples'][i]
        rchem_pot = num_staples * rstaple_u
        rpot = enes[i] + rchem_pot
        reduced_potentials.append(rpot)

    return np.array(reduced_potentials)


def subsample_independent_config_set(win_rpots):
    print('Window, configs, t0, g,   Neff')
    win_subsample_indices = []
    for i, rpots in enumerate(win_rpots):

        # t is start of equilbrated subset, g is statistical innefficiency,
        # Neff is effective sample number
        t, g, Neff = timeseries.detectEquilibration(np.array(rpots))
        print('{:<7} {:<8} {:<3} {:<4.1f} {:<.1f}'.format(i, len(rpots), t, g, Neff))
        prod_indices = timeseries.subsampleCorrelatedData(rpots[t:], g=g)
        indices = [i + t for i in prod_indices]
        #indices = list(range(len(rpots)))
        win_subsample_indices.append(indices)

    return win_subsample_indices


    win_uncorrelated_enes = []
    win_uncorrelated_ops = []

def create_uncorrelated_concatenation(wins, win_subsample_indices, win_obvs):
    subsample_indices = win_subsample_indices[0]
    win_uncorrelated_obvs = np.array(win_obvs[0])[subsample_indices]
    for i in range(1, len(wins)):
        subsample_indices = win_subsample_indices[i]
        win_subsampled_obvs = np.array(win_obvs[i])[subsample_indices]
        win_uncorrelated_obvs = np.concatenate([win_uncorrelated_obvs,
            win_subsampled_obvs])

    return win_uncorrelated_obvs

def create_uncorrelated_ops_concatenation(wins, win_subsample_indices, win_ops):
    win_uncorrelated_ops = {}
    tags = win_ops[0].keys()
    subsample_indices = win_subsample_indices[0]
    for tag in tags:
        win_uncorrelated_ops[tag] = np.array(win_ops[0][tag])[subsample_indices]

    for i in range(1, len(wins)):
        subsample_indices = win_subsample_indices[i]
        for tag in tags:
            win_subsampled_ops = np.array(win_ops[i][tag])[subsample_indices]
            win_uncorrelated_ops[tag] = np.concatenate(
                    [win_uncorrelated_ops[tag], win_subsampled_ops])

    return win_uncorrelated_ops

def calc_uncorrelated_rpots(wins, win_uncorrelated_enes, win_uncorrelated_ops,
        win_biases, rstaple_u, tags, min_bias, slope):

    uncorrelated_rpots = []
    for i in range(len(wins)):
        rpots = calc_reduced_potentials(wins[i], win_uncorrelated_enes,
                win_uncorrelated_ops, win_biases[i], rstaple_u, tags, min_bias,
                slope)
        uncorrelated_rpots.append(rpots)

    return uncorrelated_rpots


def sort_and_fill_pmfs(bins, pmfs, staple_lims, domain_lims):
    bin_pmf = {bins[i]: pmfs[i] for i in range(len(bins))}
    for x in range(staple_lims[0], staple_lims[1] + 1):
        for y in range(domain_lims[0], domain_lims[1] + 1):
            if (x, y) not in bin_pmf.keys():
                bin_pmf[(x, y)] = 'nan'

    sorted_bin_pmf = sorted(bin_pmf.items(), key=itemgetter(0))
    bins = []
    pmfs = []
    for point, pmf in sorted_bin_pmf:
        bins.append(point)
        pmfs.append(pmf)

    return bins, pmfs
