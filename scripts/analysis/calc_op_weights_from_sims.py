#!/usr/bin/env python3

"""Marginalize and output simulated weights for plotting"""

import argparse
import numpy as np

from origamipy.averaging import *
from origamipy.pgfplots import write_pgf_with_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filebase', type=str, help='Filebase')
    parser.add_argument('run', type=int, help='Run number')
    parser.add_argument('reps', type=int, help='Numer of reps')
    parser.add_argument('burn_in', type=int, help='Number of burn in steps')
    parser.add_argument('input_dir', type=str, help='Directory of inputs')
    parser.add_argument('output_dir', type=str, help='Directory to output to')
    parser.add_argument('temp', type=int, help='Temperature')
    parser.add_argument('--tags', nargs='+', type=str, help='Order parameter tags')

    args = parser.parse_args()

    filebase = args.filebase
    run = args.run
    reps = args.reps
    burn_in = args.burn_in
    input_dir = args.input_dir
    output_dir = args.output_dir
    temp = args.temp
    tags = args.tags

    # First calculate the freqeuency of occurance of each order parameter for
    # each replica
    filebase_run = filebase + '_run-{}'.format(run)
    op_weights_per_rep = calc_rep_op_weights(tags, input_dir + '/' + filebase_run,
            burn_in, reps, temp)
    filebase_run = '{}_run-{}'.format(filebase, run)

    # Calculate average op values and average pmfs
    for tag in tags:
        op, weights_per_rep = order_and_split_dictionary(op_weights_per_rep[tag])

        op_means, op_stds = calc_mean_std(weights_per_rep)
        op_filename = (output_dir + '/' + filebase_run +
                '-{}_weights.{}'.format(temp, tag))
        write_pgf_with_errors(op_filename, op, op_means, op_stds)

        pmf_means, pmf_stds = calc_pmf_with_stds(op_means, op_stds)
        pmf_filename = (output_dir + '/' + filebase_run +
                '-{}_pmfs.{}'.format(temp, tag))
        write_pgf_with_errors(pmf_filename, op, pmf_means, pmf_stds)

if __name__ == '__main__':
    main()
