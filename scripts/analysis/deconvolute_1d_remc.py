#!/usr/bin/env python3

"""Deconvolute a 1D temperature REMC simulation."""

import argparse

from origamipy import remc
from origamipy import io


def main():
    args = parse_args()
    all_params = create_exchange_params(args.temps)
    fileinfo = io.FileInfo(args.inputdir, args.outputdir, args.filebase)
    remc.deconvolute_remc_outputs(all_params, fileinfo, FILETYPES)


FILETYPES = ['ene', 'ops', 'trj', 'vcf', 'times', 'ores', 'states', 'staples',
             'staplestates']


def create_exchange_params(temps):

    # This is the order the exchange parameters are output in the exchange file
    # TODO: Have this be read from the exchange file
    all_params = []
    for temp in temps:
        all_params.append(temp)

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

    return parser.parse_args()


if __name__ == '__main__':
    main()
