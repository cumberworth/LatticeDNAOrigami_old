#!/bin/env python

"""Calculate instantanious scaffold radius of gyration."""

import argparse

import numpy as np

from origamipy import config_process
from origamipy import files

def main():
    args = parse_args()
    system_file = files.JSONStructInpFile(args.system_filename)
    traj_file = files.TxtTrajInpFile(args.traj_filename, system_file)
    rgs = config_process.calc_radius_of_gyration(traj_file)

    np.savetxt(args.out_file, rgs)

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
