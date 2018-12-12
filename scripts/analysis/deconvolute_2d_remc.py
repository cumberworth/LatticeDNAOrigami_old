#!/usr/bin/env python3

"""Deconvolute a 2D temperature-stacking multiplier simulation"""

import argparse

from origamipy import remc
from origamipy import files


def main():
    args = parse_args()
    all_params = create_2d_exchange_params(args.temps, args.stack_mults)
    fileinfo = files.FileInfo(args.inputdir, args.outputdir, args.filebase)
    remc.deconvolute_remc_outputs(all_params, fileinfo, FILETYPES)


FILETYPES = ['ene', 'ops', 'trj', 'vcf', 'times', 'ores', 'states', 'staples',
             'staplestates']


def create_2d_exchange_params(temps, stack_mults):

    # This is the order the exchange parameters are output in the exchange file
    # TODO: Have this be read from the exchange file
    all_params = []
    for temp in temps:
        for stackm in stack_mults:
            all_params.append('{}-{}'.format(temp, stackm))

    return all_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'inputdir',
            type=str,
            help='Input file directory')
    parser.add_argument(
            'outputdir',
            type=str,
            help='Output file directory')
    parser.add_argument(
            'filebase',
            type=str,
            help='Base name for files')
    parser.add_argument(
            '--temps',
            nargs='+',
            type=str,
            help='Temperatures')
    parser.add_argument(
            '--stack_mults',
            nargs='+',
            type=str,
            help='Stacking energy multipliers')

    return parser.parse_args()


if __name__ == '__main__':
    main()
