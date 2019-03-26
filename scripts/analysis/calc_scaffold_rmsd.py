#!/bin/env python

"""Calculate scaffold RMSD relative to a given configuration."""

import argparse

import numpy as np

from origamipy import config_process
from origamipy import files

def main():
    args = parse_args()
    system_file = files.JSONStructInpFile(args.system_filename)
    traj_file = files.TxtTrajInpFile(args.traj_filename, system_file)
    ref_ext = args.ref_filename.split('.')[-1]
    if ref_ext == 'json':
        ref_file = files.JSONStructInpFile(args.ref_filename)
    elif ref_ext == 'trj':
        ref_file = files.TxtTrajInpFile(args.ref_filename, system_file)

    ref_config = np.array(ref_file.chains(args.step))
    ref_positions = config_process.center_on_origin(ref_config)
    aligned_positions, rmsds = config_process.align_positions(traj_file,
            ref_config)

    np.savetxt(args.outfile, rmsds)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'system_filename',
            type=str,
            help='System file')
    parser.add_argument(
            'traj_filename',
            type=str,
            help='Configurations file to calculate RMSD on')
    parser.add_argument(
            'ref_filename',
            type=str,
            help='Configurations file containing reference')
    parser.add_argument(
            'step',
            type=int,
            help='Step containing reference configuration')
    parser.add_argument(
            'outfile',
            type=str,
            help='Output filename')

    return parser.parse_args()


if __name__ == '__main__':
    main()
