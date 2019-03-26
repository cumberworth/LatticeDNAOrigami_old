#!/usr/bin/env python

"""Carry out standard MBAR analysis on 1D REMC simulation output.

The exchange variable is assumed to be temperature.
in that order.
"""

import argparse

from origamipy import conditions
from origamipy import biases
from origamipy import outputs
from origamipy import decorrelate
from origamipy import mbar_wrapper


def main():
    args = parse_args()
    fileformatter = construct_fileformatter()
    all_conditions = construct_conditions(args, fileformatter)
    inp_filebase = create_input_filepathbase(args)
    sim_collections = outputs.create_sim_collections(inp_filebase,
            all_conditions, args.reps)
    decor_outs = decorrelate.DecorrelatedOutputs(sim_collections, all_conditions)
    decor_outs.read_decors_from_files()

    mbarw = mbar_wrapper.MBARWrapper(decor_outs)
    mbarw.perform_mbar()

    out_filebase = create_output_filepathbase(args)
    mbarw.calc_all_expectations(out_filebase)
    mbarw.calc_1d_lfes(all_conditions, out_filebase)
    mbarw.calc_specified_2d_lfes(parse_tag_pairs(args.tag_pairs),
            all_conditions, out_filebase)


def parse_tag_pairs(tag_pairs):
    return [tuple(tag_pair.split(',')) for tag_pair in tag_pairs]


def construct_conditions(args, fileformatter):
    conditions_map = {'temp': args.temps,
                      'staple_m': [args.staple_m],
                      'bias': [biases.NoBias()]}

    return conditions.AllSimConditions(conditions_map, fileformatter)


def construct_fileformatter():
    specs = [conditions.ConditionsFileformatSpec('temp', '{:d}')]
    return conditions.ConditionsFileformatter(specs)


def create_input_filepathbase(args):
    return '{}/{}'.format(args.input_dir, args.filebase)


def create_output_filepathbase(args):
    return '{}/{}'.format(args.output_dir, args.filebase)


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
        '--reps',
        nargs='+',
        type=int,
        help='Reps (leave empty for all available)')
    parser.add_argument(
            '--temps',
            nargs='+',
            type=int,
            help='Temperatures')
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
