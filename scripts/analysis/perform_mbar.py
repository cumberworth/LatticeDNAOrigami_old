#!/usr/bin/env python

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
    uncorrelated_dists = uncorrelated_ops[:, -3].astype(int)

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
    dists_bins = set(uncorrelated_dists)
    dists_to_bin = {j: i for i, j in enumerate(dists_bins)}
    uncor_dists_bin_is = [dists_to_bin[i] for i in uncorrelated_dists]
    uncor_dists_bin_is = np.array(uncor_dists_bin_is)
    pmf_dists, var_pmf_dists = origami_mbar.computePMF(
            no_bias_uncorrelated_rpots, uncor_dists_bin_is,
            len(dists_bins))
    write_pgf_file(dists_bins, pmf_dists, np.sqrt(var_pmf_dists),
            filebase + '_pmfs.dists')

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

 
if __name__ == '__main__':
    main()
