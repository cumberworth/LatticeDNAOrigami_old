#!/usr/bin/env pythonm

"""Take output from op files and average over replicas across range of Ts

Output in pgf compatible format
"""

import argparse
import numpy as np
import pdb
import os
import sys

from origamipy.averaging import calc_mean_ops
from origamipy.op_process import read_ops_from_file
from origamipy.pgfplots import write_pgf_with_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filebase', type=str, help='Filebase')
    parser.add_argument('run', type=int, help='Run number')
    parser.add_argument('reps', type=int, help='Numer of reps')
    parser.add_argument('burn_in', type=int, help='Number of burn in steps')
    parser.add_argument('input_dir', type=str, help='Directory of inputs')
    parser.add_argument('output_dir', type=str, help='Directory to output to')
    parser.add_argument('--temps', nargs='+', type=int, help='Temperatures')
    parser.add_argument('--tags', nargs='+', type=str, help='Order parameter tags')

    args = parser.parse_args()

    filebase = args.filebase
    run = args.run
    reps = args.reps
    burn_in = args.burn_in
    input_dir = args.input_dir
    output_dir = args.output_dir
    tags = args.tags
    temps = args.temps

    op_means = {tag: [] for tag in tags}
    op_stds = {tag: [] for tag in tags}
    for temp in temps:
        op_reps = {tag: np.zeros(reps) for tag in tags}
        for rep in range(reps):
            filename = '{}/{}_run-{}_rep-{}-{}.ops'.format(input_dir, filebase,
                    run, rep, temp)
            ops = read_ops_from_file(filename, tags, burn_in)
            mean_ops = calc_mean_ops(tags, ops)
            for tag, mean in mean_ops.items():
                op_reps[tag][rep] = mean

        for tag, means in op_reps.items():
            op_means[tag].append(means.mean())
            op_stds[tag].append(means.std())

    # Write to file
    for tag in tags:
        filename = '{}/{}_run-{}_{}_aves.dat'.format(output_dir, filebase, run,
                tag)
        write_pgf_with_errors(filename, temps, op_means[tag], op_stds[tag])


if __name__ == '__main__':
    main()
