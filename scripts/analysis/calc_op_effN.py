#!/usr/bin/env pythonm

"""Calculate effective sample number for each temperature

Output in pgf compatible format
"""

import argparse
import numpy as np
import pdb
import os
import sys

from origamipy.averaging import calc_mean_ops
from origamipy.averaging import calc_effN_ops
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

    op_effN_means = {tag: [] for tag in tags}
    op_effN_stds = {tag: [] for tag in tags}
    op_g_means = {tag: [] for tag in tags}
    op_g_stds = {tag: [] for tag in tags}
    op_t_means = {tag: [] for tag in tags}
    op_t_stds = {tag: [] for tag in tags}
    for temp in temps:
        op_rep_effNs = {tag: np.zeros(reps) for tag in tags}
        op_rep_gs = {tag: np.zeros(reps) for tag in tags}
        op_rep_ts = {tag: np.zeros(reps) for tag in tags}
        for rep in range(reps):
            filename = '{}/{}_run-{}_rep-{}-{}.ops'.format(input_dir, filebase,
                    run, rep, temp)
            ops = read_ops_from_file(filename, tags, burn_in)
            t_g_effN_ops = calc_effN_ops(tags, ops)
            for tag, t_g_effN in t_g_effN_ops.items():
                op_rep_ts[tag][rep] = t_g_effN[0]
                op_rep_gs[tag][rep] = t_g_effN[1]
                op_rep_effNs[tag][rep] = t_g_effN[2]

        for tag, effNs in op_rep_effNs.items():
            op_effN_means[tag].append(effNs.mean())
            op_effN_stds[tag].append(effNs.std())
        for tag, gs in op_rep_gs.items():
            op_g_means[tag].append(gs.mean())
            op_g_stds[tag].append(gs.std())
        for tag, ts in op_rep_ts.items():
            op_t_means[tag].append(ts.mean())
            op_t_stds[tag].append(ts.std())

    # Write to file
    for tag in tags:
        filename = '{}/{}_run-{}_{}_effNs.dat'.format(output_dir, filebase, run,
                tag)
        write_pgf_with_errors(filename, temps, op_effN_means[tag],
                op_effN_stds[tag])
        filename = '{}/{}_run-{}_{}_gs.dat'.format(output_dir, filebase, run,
                tag)
        write_pgf_with_errors(filename, temps, op_g_means[tag], op_g_stds[tag])
        filename = '{}/{}_run-{}_{}_ts.dat'.format(output_dir, filebase, run,
                tag)
        write_pgf_with_errors(filename, temps, op_t_means[tag], op_t_stds[tag])


if __name__ == '__main__':
    main()
