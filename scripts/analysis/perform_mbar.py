#!/usr/bin/env python

"""Carry out MBAR analysis on given set of US simulation trajectories"""

import argparse
import json
import math

import scipy.constants
import numpy as np
import pymbar.mbar as mbar
from operator import itemgetter

from origamipy.mbar import *
from origamipy.pgfplots import *
from origamipy.us_process import *

LATTICE_SITE_VOLUME = 4e-28
MIN_BIAS = 100
SLOPE = 10

def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filebase', type=str, help='Filebase simulation output files')
    parser.add_argument('input_dir', type=str, help='Directory of inputs')
    parser.add_argument('output_dir', type=str, help='Directory to output to')
    parser.add_argument('windows_file', type=str, help='Windows file')
    parser.add_argument('temp', type=float, help='System temperature')
    parser.add_argument('staple_M', type=float, help='Staple molarity (mol/V)')
    parser.add_argument('--tags', nargs='+', type=str, help='Order parameter tags')

    args = parser.parse_args()
    
    filebase = args.filebase
    inputs_dir = args.input_dir
    output_dir = args.output_dir
    wins_filename = args.windows_file
    temp = args.temp
    staple_M = args.staple_M
    tags = args.tags

    rstaple_u = calc_rstaple_u(staple_M, LATTICE_SITE_VOLUME)

    # Read and prepare inputs
    bias_tags, wins = read_windows_file(wins_filename)
    win_filebases = create_window_filebases(wins, inputs_dir + '/' + filebase)
    #win_enes = read_win_energies(win_filebases)
    win_enes = read_win_energies_from_log(win_filebases)
    win_ops = read_win_order_params(win_filebases, tags)
    win_biases = read_win_grid_biases(wins, win_filebases)
    win_correlated_rpots = calc_correlated_rpots(wins, win_enes, win_ops,
            win_biases, rstaple_u, bias_tags, MIN_BIAS, SLOPE)
    win_subsample_indices = subsample_independent_config_set(
            win_correlated_rpots)
    uncorrelated_enes = create_uncorrelated_concatenation(wins,
            win_subsample_indices, win_enes)
    uncorrelated_ops = create_uncorrelated_ops_concatenation(wins,
            win_subsample_indices, win_ops)
    win_uncorrelated_rpots = calc_uncorrelated_rpots(wins,
            uncorrelated_enes, uncorrelated_ops, win_biases, rstaple_u,
            bias_tags, MIN_BIAS, SLOPE)
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

    for i, tag in enumerate(tags):

        # Expectations
        ave_ops, var_ops = origami_mbar.computeExpectations(
                uncorrelated_ops[tag], no_bias_uncorrelated_rpots)
        std_ops = np.sqrt(var_ops)
        print('Average of {}: {:.2f}+-{:.2f}'.format(tag, ave_ops[0],
            std_ops[0]))

        # 1D PMFS (make a function for this)
        op_bins = set(uncorrelated_ops[tag])
        op_to_bin = {j: i for i, j in enumerate(op_bins)}
        uncor_op_bin_is = [op_to_bin[i] for i in uncorrelated_ops[tag]]
        uncor_op_bin_is = np.array(uncor_op_bin_is)
        pmf_ops, var_pmf_ops = origami_mbar.computePMF(
                no_bias_uncorrelated_rpots, uncor_op_bin_is, len(op_bins))
        filename = output_dir + '/' + filebase + '_pmfs.{}'.format(tag)
        write_pgf_with_errors(filename, op_bins, pmf_ops, np.sqrt(var_pmf_ops))

        for tag2 in tags[i + 1:]:

            # 2D PMFS
            uncorrelated_op_pairs = list(zip(uncorrelated_ops[tag],
                    uncorrelated_ops[tag2]))
            op_pair_bins = list(set(uncorrelated_op_pairs))
            op_pair_to_bin = {j: i for i, j in enumerate(op_pair_bins)}
            uncor_op_pair_bin_is = [op_pair_to_bin[i] for i in
                    uncorrelated_op_pairs]
            uncor_op_pair_bin_is = np.array(uncor_op_pair_bin_is)
            pmf_op_pairs, var_pmf_op_pairs= origami_mbar.computePMF(
                    no_bias_uncorrelated_rpots, uncor_op_pair_bin_is,
                    len(op_pair_bins))
            op1_lims = [0, max(op_pair_bins[0])]
            op2_lims = [0, max(op_pair_bins[1])]
            sd_bins, pmf_sds = sort_and_fill_pmfs(op_pair_bins, pmf_op_pairs,
                    op1_lims, op2_lims)
            filename = output_dir + '/' + filebase + '_pmfs.{}-{}'.format(tag,
                    tag2)
            write_2d_pgf(filename, op_pair_bins, pmf_op_pairs)

 
if __name__ == '__main__':
    main()
