#!/bin/env python

"""Calculate scaffold end-to-end distance."""

import argparse

import numpy as np

from origamipy import config_process
from origamipy import files

def main():
    args = parse_args()
    system_file = files.JSONStructInpFile(args.system_filename)
    traj_file = files.TxtTrajInpFile(args.traj_filename, system_file)
    dists = config_process.calc_end_to_end_dists(traj_file)

    np.savetxt(args.out_file, dists)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'system_filename',
            type=str,
            help='System file')
    parser.add_argument(
            'traj_filename',
            type=str,
            help='Configurations file to perform calculation on')
    parser.add_argument(
            'out_file',
            type=str,
            help='Output filename')

    return parser.parse_args()


if __name__ == '__main__':
    main()
