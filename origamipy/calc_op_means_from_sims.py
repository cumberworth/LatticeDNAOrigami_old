#!/usr/bin/env pythonm

"""Take output from op files and average over replicas across range of Ts

Output in pgf compatible format
"""

import numpy as np
import pdb
import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filebase', type=str, help='Filebase')
    parser.add_argument('run', type=int, help='Run number')
    parser.add_argument('reps', type=int, help='Numer of reps')
    parser.add_argument('temps', type=str, help='Temperatures')
    parser.add_argument('tags', type=str, help='Order parameter tags')
    parser.add_argument('burn_in', type=int, help='Number of burn in steps')

    args = parser.parse_args()

    filebase = args.filebase
    run = args.run
    reps = args.reps
    burn_in = args.burn_in
    temps = [float(n) for n in args.temps.split()]

    op_means = {tag: [] for tag in tags}
    op_stds = {}
    for temp in temps:
        op_reps = {tag: np.zeros(reps) for tag in tags}
        for rep in range(reps):
            filename = '{}_run-{}_rep-{}-{}.counts'.format(filebase, run, rep, temp)
            ops = read_ops_from_file(filename)
            mean_ops = calc_mean_ops(tags, ops)
            for tag, mean in mean_ops.items():
                op_reps[tag][rep] = mean

        for tag, means in op_reps:
            op_means[tag].append(means.mean())
            op_stds[tag].append(means.std())

`   # Write to file
    for tag in tags:
        filename = '{}_run-{}_{}_aves.dat'.format(filebase, run, tag)
        write_pgf_file(temps, op_means[tag], op_stds[tag], filename)


if __name__ == '__main__':
    main()
