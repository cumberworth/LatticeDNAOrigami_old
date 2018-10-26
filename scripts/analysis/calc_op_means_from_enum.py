#!/usr/bin/env pythonm

"""Take output from count matrices and average over rows or columns

Output in format useable by PGF plots
"""

import argparse
import pdb
import sys

import numpy as np

from origamipy.averaging import calc_mean_ops_from_weights
from origamipy.averaging import marginalize_ops
from origamipy.op_process import read_weights_from_file
from origamipy.pgfplots import write_pgf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filebase', type=str, help='Filebase')
    parser.add_argument('input_dir', type=str, help='Directory of inputs')
    parser.add_argument('output_dir', type=str, help='Directory to output to')
    parser.add_argument('--temps', nargs='+', type=int, help='Temperatures')
    parser.add_argument('--tags', nargs='+', type=str, help='Order parameter tags')

    args = parser.parse_args()

    filebase = args.filebase
    input_dir = args.input_dir
    output_dir = args.output_dir
    temps = args.temps
    tags = args.tags

    op_means = {tag: [] for tag in tags}
    for temp in temps:
        filename = '{}/{}_temp-{}.weights'.format(input_dir, filebase, temp)
        header_tags, ops_weights = read_weights_from_file(filename)
        for tag in tags:
            weights = marginalize_ops(ops_weights, header_tags, tag)
            op_means[tag].append(calc_mean_ops_from_weights(weights))

    for tag, means in op_means.items():
        filename = '{}/{}_{}_aves.dat'.format(output_dir, filebase, tag)
        write_pgf(filename, temps, means)


if __name__ == '__main__':
    main()
