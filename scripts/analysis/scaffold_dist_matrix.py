#!/usr/bin/env python3

"""Calculate internal distance matrix from a trajectory"""

import argparse
import sys
sys.path.insert(0, '../../src/lattice_origami_domains')
from lattice_dna_origami.origami_io import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('system_file', type=str, help='System file')
    parser.add_argument('traj_file', type=str, help='Trajectory file')
    parser.add_argument('out_filebase', type=str, help='Output file')
    parser.add_argument('skip', type=int, help='Saved interval')
    parser.add_argument('skipo', type=int, help='Read interval')

    args = parser.parse_args()
    system_filename = args.system_file
    traj_filename = args.traj_file
    out_filebase = args.out_filebase
    skip = args.skip
    skipo = args.skipo

    system_file = JSONInputFile(system_filename)
    traj_file = PlainTextTrajFile(traj_filename, system_file)

    dist_matrices = calc_dist_matrices(traj_file, skip, skipo)

    dist_sums = np.sum(dist_matrices, axis=1)
    steps = [(i + 1) * skip * skipo for i in range(len(dist_sums))]
    write_pgf_file(steps, dist_sums, out_filebase + '_dist_sums.dat')

    ref_config = np.array(system_file.chains(0)[0]['positions'])
    ref_dists = calc_dist_matrix(ref_config)
    dist_rmsds = calc_rmsds(ref_dists, dist_matrices)
    write_pgf_file(steps, dist_rmsds, out_filebase + '_dist_rmsds.dat')


if __name__ == '__main__':
    main()
