#!/usr/bin/env python

"""Perform decorrelation on 1D REMC simulation output.

The exchange variables are assumed to be temperature and stacking multiplier,
in that order.
"""

import argparse

from origamipy import biases
from origamipy import conditions
from origamipy import decorrelate
from origamipy import files
from origamipy import outputs
from origamipy import utility


def main():
    args = parse_args()
    system_file = files.JSONStructInpFile(args.system_filename)
    staple_lengths = utility.calc_staple_lengths(system_file)
    fileformatter = construct_fileformatter()
    all_conditions = conditions.construct_remc_conditions(
        args.temps, args.staple_m, fileformatter, staple_lengths)
    inp_filebase = f'{args.input_dir}/{args.filebase}'
    sim_collections = []
    for rep in range(args.reps):
        rep_sim_collections = outputs.create_sim_collections(
            inp_filebase, all_conditions, rep)
        sim_collections.append(rep_sim_collections)

    decor_outs = decorrelate.DecorrelatedOutputs(
        sim_collections, rep_conditions_equal=True)
    decor_outs.perform_decorrelation(args.skip)
    out_filebase = '{}/{}'.format(args.output_dir, args.filebase)
    decor_outs.apply_masks(out_filebase)
    decor_outs.write_decors_to_files(out_filebase)


def construct_fileformatter():
    specs = [conditions.ConditionsFileformatSpec('temp', '{}')]
    return conditions.ConditionsFileformatter(specs)


def construct_conditions(args, fileformatter, system_file):
    conditions_map = {'temp': args.temps,
                      'staple_m': [args.staple_m],
                      'bias': [biases.NoBias()]}

    return conditions.AllSimConditions(conditions_map, fileformatter, system_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'system_filename',
        type=str,
        help='System file')
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
        'skip',
        type=int,
        help='Number of steps to skip')
    parser.add_argument(
        'reps',
        type=int,
        help='Number of reps')
    parser.add_argument(
        '--temps',
        nargs='+',
        type=str,
        help='Temperatures')
    parser.add_argument(
        '--stack_mults',
        nargs='+',
        type=float,
        help='Stacking energy multipliers')

    return parser.parse_args()


if __name__ == '__main__':
    main()
