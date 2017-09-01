#!/usr/bin/env pythonm

"""Take output from count matrices and average over rows or columns

Output in format useable by PGF plots
"""

import argparse
import pdb
import sys

import numpy as np


TEMPS = [330, 335, 340, 345, 350, 355, 360]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filebase', type=str, help='Filebase')
    args = parser.parse_args()
    filebase = args.filebase

    domain_aves = [] # Fully bound domains
    misdomain_aves = [] # Misbound domains
    staple_aves = []

    # Iterate over temperatures
    for temp in TEMPS:
        filename = '{}-{}_weights.domains'.format(filebase, temp)
        domain_weights = np.loadtxt(filename, skiprows=1)
        filename = '{}-{}_weights.staples'.format(filebase, temp)
        staple_weights = np.loadtxt(filename, skiprows=1)

        # Domain aves
        domain_ave = 0
        for domains, weight in domain_weights:
            domain_ave += domains * weight

        domain_aves.append(domain_ave)

        # Staple aves
        staple_ave = 0
        for staples, weight in staple_weights:
            staple_ave += staples * weight

        staple_aves.append(staple_ave)

    write_pgf_file(TEMPS, domain_aves, '{}_domain_aves.dat'.format(filebase))
    write_pgf_file(TEMPS, staple_aves, '{}_staple_aves.dat'.format(filebase))


def write_pgf_file(xdata, ydata, filename):
    with open(filename, 'w') as inp:
        inp.write('x y\n')
        for x, y in zip(xdata, ydata):
            inp.write('{} {}\n'.format(x, y))


if __name__ == '__main__':
    main()
