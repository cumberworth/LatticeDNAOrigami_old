#!/bin/env python

"""Calculate scaffold RMSD relative to a given configuration."""

import math
import argparse
import sys

import numpy as np
import scipy.constants

sys.path.insert(0, '../../src/lattice_origami_domains')

from lattice_dna_origami.origami_io import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('system_filename', type=str, help='System file')
    parser.add_argument('traj_filename', type=str, help='Configurations file to calculate RMSD on')
    parser.add_argument('skip', type=int, help='Saved interval')
    parser.add_argument('skipo', type=int, help='Read interval')
    parser.add_argument('ref_filename', type=str, help='Configurations file containing reference ')
    parser.add_argument('step', type=int, help='Step containing reference configuration')
    parser.add_argument('outfile', type=str, help='Output filename')

    args = parser.parse_args()
    system_filename = args.system_filename
    traj_filename = args.traj_filename
    skip = args.skip
    skipo = args.skipo
    ref_filename = args.ref_filename
    step = args.step
    outfile = args.outfile

    system_file = JSONInputFile(system_filename)
    traj_file = PlainTextTrajFile(traj_filename, system_file)
    ref_file = PlainTextTrajFile(ref_filename, system_file)
    #ref = np.array(ref_file.chains(step)[0]['positions'])[3:-4]
    ref = np.array(ref_file.chains(step)[0]['positions'])
    aligned_configs, rmsds, steps = align_configs(traj_file, ref, skip, skipo)
    output_row_major(outfile, rmsds, steps)

if __name__ == '__main__':
    main()
