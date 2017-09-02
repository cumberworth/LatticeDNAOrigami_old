"""Functions for output data in PGF compatable format"""

import numpy as np


def write_pgf(filename, xdata, ydata):
    with open(filename, 'w') as inp:
        inp.write('x y\n')
        for x, y in zip(xdata, ydata):
            inp.write('{} {}\n'.format(x, y))


def read_pgf(filename):
    return np.loadtxt(filename, skiprows=1)


def write_pgf_with_errors(filename, xdata, ydata, errors):
    with open(filename, 'w') as inp:
        inp.write('x y error\n')
        for x, y, e in zip(xdata, ydata, errors):
            inp.write('{} {} {}\n'.format(x, y, e))


def write_2d_pgf(filename, xydata, zdata):
    with open(filename, 'w') as inp:
        inp.write('x y DF\n')
        for xy, z in zip(xydata, zdata):
            inp.write('{} {} {}\n'.format(xy[0], xy[1], z))
