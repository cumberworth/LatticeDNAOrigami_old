#!/usr/bin/env python3

"""Marginalize and output simulated weights for plotting"""

import argparse
import numpy as np

from origamipy.averaging import *
from origamipy.pgfplots import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filebase', type=str, help='Filebase')
    parser.add_argument('run', type=int, help='Run number')
    parser.add_argument('reps', type=int, help='Numer of reps')
    parser.add_argument('burn_in', type=int, help='Number of burn in steps')
    parser.add_argument('input_dir', type=str, help='Directory of inputs')
    parser.add_argument('output_dir', type=str, help='Directory to output to')
    parser.add_argument('temp', type=int, help='Temperature')
    parser.add_argument('tag1', type=str, help='Order parameter tag 1')
    parser.add_argument('tag2', type=str, help='Order parameter tag 2')

    args = parser.parse_args()

    filebase = args.filebase
    run = args.run
    reps = args.reps
    burn_in = args.burn_in
    input_dir = args.input_dir
    output_dir = args.output_dir
    temp = args.temp
    tag1 = args.tag1
    tag2 = args.tag2

    # First calculate the freqeuency of occurance of each order parameter for
    # each replica
    filebase_run = filebase + '_run-{}'.format(run)
    tags = [tag1, tag2]
    op_weights_per_rep = calc_rep_op_combined_weights(tags, input_dir + '/' +
            filebase_run, burn_in, reps, temp)
    filebase_run = '{}_run-{}'.format(filebase, run)

    # Calculate average op values and average pmfs
    ops, weights_per_rep = order_fill_and_split_dictionary(op_weights_per_rep)

    op_means, op_stds = calc_mean_std(weights_per_rep)
    pmf_means, pmf_stds = calc_pmf_with_stds(op_means, op_stds)
    pmf_filename = (output_dir + '/' + filebase_run +
            '-{}_{}-{}_pmfs.dat'.format(temp, tag1, tag2))
    write_2d_pgf(pmf_filename, ops, pmf_means)

if __name__ == '__main__':
    main()
