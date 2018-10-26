#!/usr/bin/env python

"""Carry out standard MBAR analysis on 2D REMC simulation output.

The exchange variables are assumed to be temperature and stacking multiplier,
in that order.
"""

import argparse

import pymbar.mbar as mbar

from origamipy import mbar
from origamipy import biases


def main():
    args = parse_args()
    conditions_map = construct_conditions_map(args)
    fileformatter = construct_fileformatter()
    all_conditions = mbar.AllSimConditions(conditions_map, fileformatter)
    filepathbase = create_filepathbase(args)
    sims_collection = mbar.MultiStateSimCollection(filepathbase, all_conditions)

    sims_collection.perform_decorrelation() #return?
    sims_collection.perform_mbar()
    #sims_collection.calculate_expectations(tags?) #return?


def construct_conditions_map(args):
    stack_biases = []
    for stack_mult in args.stack_mults:
        stack_bias = biases.StackingBias(args.stack_ene, stack_mult)
        stack_biases.append(stack_bias)

    conditions_map = {'temp': args.temps,
                     'staple_m': [args.staple_m],
                     'bias': stack_biases}

    return conditions_map


def construct_fileformatter():
    spec = mbar.ConditionsFileformatSpec(('temp', '{:d}'), ('bias', '{:.1f}'))
    return mbar.ConditionsFileformatter(spec)


def create_filepathbase(args):
    return '{}/{}'.format(args.input_dir, args.filebase)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'filebase',
            type=str,
            help='Base name for files')
    parser.add_argument(
            'input_dir',
            type=str,
            help='Directory of inputs')
    parser.add_argument(
            'output_dir',
            type=str,
            help='Directory to output to')
    parser.add_argument(
            'staple_m',
            type=float,
            help='Staple molarity (mol/V)')
    parser.add_argument(
            'stack_ene',
            type=float,
            help='Stacking energy (kb K)')
    parser.add_argument(
            '--temps',
            nargs='+',
            type=int,
            help='Temperatures')
    parser.add_argument(
            '--stack_mults',
            nargs='+',
            type=float,
            help='Stacking energy multipliers')
    parser.add_argument('--tags',
            nargs='+',
            type=str,
            help='Order parameter tags')
    parser.add_argument('--tag_pairs',
            nargs='+',
            type=str,
            help='Tags to calculate 2D pmf for (comma delim)')

    return parser.parse_args()


if __name__ == '__main__':
    main()
