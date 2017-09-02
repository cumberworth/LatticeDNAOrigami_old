#!/usr/bin/env python3

"""Marginalize and output simulated and enumerated weights for plotting"""

import argparse
import pickle
import numpy as np
from operator import itemgetter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filebase', type=str, help='Filebase')
    parser.add_argument('run', type=int, help='Run number')
    parser.add_argument('reps', type=int, help='Numer of reps')
    parser.add_argument('temp', type=int, help='Temperature')
    parser.add_argument('tags', type=str, help='Order parameter tags')

    args = parser.parse_args()

    filebase = args.filebase
    run = args.run
    reps = args.reps
    temp = args.temp
    tags = args.tags.split()

    # First calculate the freqeuency of occurance of each order parameter for
    # each replica
    op_weights_per_rep = calc_rep_op_weights(tags, filebase_run, temp)
    filebase_run = '{}_run-{}'.format(filebase, run)

    # Calculate average op values and average pmfs
    for tag in tags:
        op, weights_per_rep = order_weights(op_weights_per_rep[tag])

        op_means, op_stds = calc_mean_std(weights_per_rep)
        op_filename = filebase_run + '-{}_weights.{}'.format(temp, tag)
        # Output to plottable format

        pmf_means, pmf_stds = calc_mean_std_pmf(weights_per_rep)
        pmf_filename = filebase_run + '-{}_pmfs.{}'.format(temp, tag)
        # Output to plottable format

if __name__ == '__main__':
    main()
