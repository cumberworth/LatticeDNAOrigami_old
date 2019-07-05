#!/usr/bin/env python

"""Check REMC tempatures have converged to within give threshold"""

import argparse
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import minimize

from origamipy import files


def main():
    args = parse_args()
    error = ((np.array(args.old_temps) - np.array(args.new_temps))**2).sum()
    ave_error = error / len(args.old_temps)
    if ave_error < args.threshold_error:
        print(1)
    else:
        print(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'threshold_error',
            type=float,
            help='Per temperature error cutoff')
    parser.add_argument(
            '--old_temps',
            nargs='+',
            type=float,
            help='Old temperatures')
    parser.add_argument(
            '--new_temps',
            nargs='+',
            type=float,
            help='New temperatures')

    return parser.parse_args()


if __name__ == '__main__':
    main()
