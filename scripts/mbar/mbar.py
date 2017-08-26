#!/usr/bin/env python3

"""Carry out MBAR analysis on given set of US simulation trajectories"""

import argparse
import json
import math

import scipy.constants
import numpy as np
import pymbar.timeseries as timeseries
import pymbar.mbar as mbar
from operator import itemgetter

WELL_BIAS = 0
OUTSIDE_BIAS = 99
LATTICE_SITE_VOLUME = 4e-28

# This has gotten too long
def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filebase', type=str, help='Filebase simulation output files')
    parser.add_argument('windows_file', type=str, help='Windows file')
    parser.add_argument('temp', type=float, help='System temperature')
    parser.add_argument('staple_M', type=float, help='Staple molarity (mol/V)')

    args = parser.parse_args()
    
    filebase = args.filebase
    wins_filename = args.windows_file
    temp = args.temp
    staple_M = args.staple_M

    rstaple_u = calc_rstaple_u(staple_M, LATTICE_SITE_VOLUME)

    # Read and prepare inputs
    wins = read_windows(wins_filename)
    win_filebases = create_window_filebases(wins, filebase)
    win_enes = read_energies(win_filebases)
    win_ops = read_order_params(win_filebases)
    win_biases = read_grid_biases(wins, win_filebases)
    win_correlated_rpots = calc_correlated_rpots(wins, win_enes, win_ops,
            win_biases, rstaple_u)
    win_subsample_indices = subsample_independent_config_set(win_correlated_rpots)
    uncorrelated_enes = create_uncorrelated_concatenation(wins,
            win_subsample_indices, win_enes)
    uncorrelated_ops = create_uncorrelated_concatenation(wins,
            win_subsample_indices, win_ops)
    win_uncorrelated_rpots = calc_uncorrelated_rpots(wins,
            uncorrelated_enes, uncorrelated_ops, win_biases, rstaple_u)
    win_num_configs = [len(indices) for indices in win_subsample_indices]

    # Add no bias potential with no samples
    #no_bias_uncorrelated_rpots = calc_no_bias_reduced_potentials(
    #        uncorrelated_enes, uncorrelated_ops, rstaple_u)
    #win_uncorrelated_rpots.append(no_bias_uncorrelated_rpots)
    #win_num_configs.append(0)

    # MBAR analysis
    origami_mbar = mbar.MBAR(win_uncorrelated_rpots, win_num_configs)
    no_bias_uncorrelated_rpots = calc_no_bias_reduced_potentials(
            uncorrelated_enes, uncorrelated_ops, rstaple_u)
    uncorrelated_domains = uncorrelated_ops[:, -1].astype(int)
    uncorrelated_staples = uncorrelated_ops[:, -2].astype(int)

    # Expectations
    ave_staples, var_staples = origami_mbar.computeExpectations(
            uncorrelated_staples, no_bias_uncorrelated_rpots)
    std_staples = np.sqrt(var_staples)
    print('Average number of bound staples: {:.2f}+-{:.2f}'.format(ave_staples[0],
        std_staples[0]))
    ave_domains, var_domains = origami_mbar.computeExpectations(
            uncorrelated_domains, no_bias_uncorrelated_rpots)
    std_domains = np.sqrt(var_domains)
    print('Average number of fully bound domain pairs: {:.2f}+-{:.2f}'.format(
        ave_domains[0], std_domains[0]))

    # 1D PMFS (make a function for this)
    staple_bins = set(uncorrelated_staples)
    staple_to_bin = {j: i for i, j in enumerate(staple_bins)}
    uncor_staple_bin_is = [staple_to_bin[i] for i in uncorrelated_staples]
    uncor_staple_bin_is = np.array(uncor_staple_bin_is)
    pmf_staples, var_pmf_staples = origami_mbar.computePMF(
            no_bias_uncorrelated_rpots, uncor_staple_bin_is,
            len(staple_bins))
    write_pgf_file(staple_bins, pmf_staples, np.sqrt(var_pmf_staples),
            filebase + '_pmfs.staples')
    domain_bins = set(uncorrelated_domains)
    domain_to_bin = {j: i for i, j in enumerate(domain_bins)}
    uncor_domain_bin_is = [domain_to_bin[i] for i in uncorrelated_domains]
    uncor_domain_bin_is = np.array(uncor_domain_bin_is)
    pmf_domains, var_pmf_domains = origami_mbar.computePMF(
            no_bias_uncorrelated_rpots, uncor_domain_bin_is,
            len(domain_bins))
    write_pgf_file(domain_bins, pmf_domains, np.sqrt(var_pmf_domains),
            filebase + '_pmfs.domains')

    # 2D PMFS
    uncorrelated_sds = uncorrelated_ops[:, -2:].astype(int).tolist()
    uncor_sds = [(i[0], i[1]) for i in uncorrelated_sds]
    sd_bins = list(set(uncor_sds))
    sd_to_bin = {j: i for i, j in enumerate(sd_bins)}
    uncor_sd_bin_is = [sd_to_bin[i] for i in uncor_sds]
    uncor_sd_bin_is = np.array(uncor_sd_bin_is)
    pmf_sds, var_pmf_sds = origami_mbar.computePMF(
            no_bias_uncorrelated_rpots, uncor_sd_bin_is,
            len(sd_bins))
    staple_lims = [0, max(staple_bins)]
    domain_lims = [0, max(domain_bins)]
    sd_bins, pmf_sds = sort_and_fill_pmfs(sd_bins, pmf_sds, staple_lims, domain_lims)
    write_2d_pgf_file(sd_bins, pmf_sds, filebase + '_pmfs.sds')

def calc_rstaple_u(staple_M, lattice_site_volume):
    """Calculate reduced staple chemical potential"""
    sites_per_litre = 1e-3 / lattice_site_volume
    rstaple_u = math.log(staple_M * scipy.constants.N_A / sites_per_litre)
    return rstaple_u 


def read_windows(wins_filename):
    """Read windows file and return list of tuple of min max tuples"""
    with open(wins_filename) as inp:
        lines = inp.readlines()

    wins = []
    for line in lines:
        mins_raw, maxs_raw = line.split(',')
        mins = tuple(map(int, mins_raw.split()))
        maxs = tuple(map(int, maxs_raw.split()))
        win = (mins, maxs)
        wins.append(win)

    return wins


def create_window_filebases(wins, filebase):
    """Create list of filebases for each window.

    Assumes MWUS simulation conventions for output filenames
    """
    win_filebases = []
    for win in wins:
        postfix = '_win'
        for win_min in win[0]:
            postfix += '-' + str(win_min)

        postfix += '-'
        for win_max in win[1]:
            postfix += '-' + str(win_max)

        win_filebases.append(filebase + postfix)

    return win_filebases


def read_energies(win_filebases):
    """Read in origami energies (without bias)"""
    win_enes = []
    for filebase in win_filebases:
        filename = filebase + '_iter-prod.ene'
        enes = np.loadtxt(filename, skiprows=1)
        win_enes.append(enes[:, 1])

    return win_enes


def read_order_params(win_filebases):
    """Read in order parameters"""
    win_ops = []
    for filebase in win_filebases:
        filename = filebase + '_iter-prod.order_params'
        ops = np.loadtxt(filename, skiprows=1)
        win_ops.append(ops[:,1:])

    return win_ops


def read_grid_biases(wins, win_filebases):
    """Read grid biases
    
    Return list of dictionaries indexed by grid point tuples
    """
    win_biases = []
    for filebase in win_filebases:
        filename = filebase + '.biases'
        biases = json.load(open(filename))
        bias_dic = {}
        for entry in biases['biases']:
            point = tuple(entry['point'])
            bias = entry['bias']
            bias_dic[point] = bias

        win_biases.append(bias_dic)

    return win_biases


def calc_correlated_rpots(wins, win_enes, win_ops, win_biases, rstaple_u):
    correlated_rpots = []
    for i in range(len(wins)):
        rpots = calc_reduced_potentials(wins[i], win_enes[i], win_ops[i],
                win_biases[i], rstaple_u)
        correlated_rpots.append(rpots)

    return correlated_rpots


def calc_reduced_potentials(win, win_enes, win_ops, win_biases, rstaple_u):
    """Calculate reduced potentials as defined in shirts2008"""
    min_num_domains = win[0][0]
    max_num_domains = win[1][0]
    min_num_staples = win[0][1]
    max_num_staples = win[1][1]
    reduced_potentials = []
    for i in range(len(win_enes)):
        num_staples = win_ops[i][-2]
        num_domains = win_ops[i][-1]
        rchem_pot = num_staples * rstaple_u
        bias = 0
        if (num_staples < min_num_staples or num_staples > max_num_staples) or (
            num_domains < min_num_domains or num_domains > max_num_domains):
            bias += OUTSIDE_BIAS

        point = (int(num_domains), int(num_staples))
        if point in win_biases:
            bias += win_biases[point]

        rpot = win_enes[i] + bias + rchem_pot
        reduced_potentials.append(rpot)

    return reduced_potentials


def calc_no_bias_reduced_potentials(enes, ops, rstaple_u):
    """Calculate reduced potentials as defined in shirts2008"""
    reduced_potentials = []
    for i in range(len(enes)):
        num_staples = ops[i][-2]
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

def calc_uncorrelated_rpots(wins, win_uncorrelated_enes, win_uncorrelated_ops,
        win_biases, rstaple_u):

    uncorrelated_rpots = []
    for i in range(len(wins)):
        rpots = calc_reduced_potentials(wins[i], win_uncorrelated_enes,
                win_uncorrelated_ops, win_biases[i], rstaple_u)
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


def write_pgf_file(xdata, ydata, errors, filename):
    with open(filename, 'w') as inp:
        inp.write('x y error\n')
        for x, y, e in zip(xdata, ydata, errors):
            inp.write('{} {} {}\n'.format(x, y, e))


def write_2d_pgf_file(xydata, zdata, filename):
    with open(filename, 'w') as inp:
        inp.write('x y DF\n')
        for xy, z in zip(xydata, zdata):
            inp.write('{} {} {}\n'.format(xy[0], xy[1], z))


if __name__ == '__main__':
    main()
