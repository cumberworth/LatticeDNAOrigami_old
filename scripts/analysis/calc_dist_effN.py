#!/usr/bin/env python3

"""Calculate internal distance matrix from a trajectory"""

import argparse

from pymbar import timeseries

from origamipy.origami_io import *
from origamipy.config_process import calc_dist_matrices
from origamipy.pgfplots import write_pgf_with_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('system_file', type=str, help='System file')
    parser.add_argument('filebase', type=str, help='Filebase')
    parser.add_argument('run', type=int, help='Run number')
    parser.add_argument('reps', type=int, help='Numer of reps')
    parser.add_argument('burn_in', type=int, help='Number of burn in steps')
    parser.add_argument('input_dir', type=str, help='Directory of inputs')
    parser.add_argument('output_dir', type=str, help='Directory to output to')
    parser.add_argument('skip', type=int, help='Saved interval')
    parser.add_argument('skipo', type=int, help='Read interval')
    parser.add_argument('--temps', nargs='+', type=int, help='Temperatures')

    args = parser.parse_args()

    system_filename = args.system_file
    filebase = args.filebase
    run = args.run
    reps = args.reps
    burn_in = args.burn_in
    input_dir = args.input_dir
    output_dir = args.output_dir
    skip = args.skip
    skipo = args.skipo
    temps = args.temps

    system_file = JSONInputFile(system_filename)
    t_means = []
    t_stds = []
    g_means = []
    g_stds = []
    effN_means = []
    effN_stds = []
    for temp in temps:
        t_reps = np.zeros(reps)
        g_reps = np.zeros(reps)
        effN_reps = np.zeros(reps)
        for rep in range(reps):
            traj_filename = '{}/{}_run-{}_rep-{}-{}.trj'.format(input_dir, filebase,
                    run, rep, temp)
            traj_file = PlainTextTrajFile(traj_filename, system_file)
            dist_matrices = calc_dist_matrices(traj_file, skip, skipo)
            dist_sums = np.sum(dist_matrices, axis=1)
            t, g, effN = timeseries.detectEquilibration(dist_sums)
            t /= len(dist_sums)
            t_reps[rep] = t
            g_reps[rep] = g
            effN_reps[rep] = effN

        t_means.append(t_reps.mean())
        t_stds.append(t_reps.std())
        g_means.append(g_reps.mean())
        g_stds.append(g_reps.std())
        effN_means.append(effN_reps.mean())
        effN_stds.append(effN_reps.std())
             
    filename = '{}/{}_run-{}_dms_ts.dat'.format(output_dir, filebase, run)
    write_pgf_with_errors(filename, temps, t_means, t_stds)
    filename = '{}/{}_run-{}_dms_gs.dat'.format(output_dir, filebase, run)
    write_pgf_with_errors(filename, temps, g_means, g_stds)
    filename = '{}/{}_run-{}_dms_effNs.dat'.format(output_dir, filebase, run)
    write_pgf_with_errors(filename, temps, effN_means, effN_stds)


if __name__ == '__main__':
    main()
