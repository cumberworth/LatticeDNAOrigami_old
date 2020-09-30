#!/usr/bin/env python

"""Extract configurations with specified order params."""

import argparse
import pdb
import sys

from origamipy import files

from origamipy import biases
from origamipy import conditions
from origamipy import files
from origamipy import outputs


def main():
    args = parse_args()
    system_file = files.JSONStructInpFile(args.system_filename)
    fileformatter = construct_fileformatter()
    all_conditions = construct_conditions(args, fileformatter, system_file)
    inp_filebase = create_input_filepathbase(args)
    sim_collections = outputs.create_sim_collections(inp_filebase,
                                                     all_conditions, args.reps)
    for sim_collection in sim_collections:
        ops_series_runs = sim_collection.get_reps_data('ops')
        filtered_steps = []
        for ops_series in ops_series_runs:
            if ops_series == 0:
                continue
            for step in range(ops_series.steps):
                filter_step = True
                for tag, value in zip(args.tags, args.values):
                    if ops_series[tag][step] != value:
                        filter_step = False
                        break

                    filtered_steps.append(step)

        if filtered_steps != []:
            print(sim_collection.conditions.fileformat)
            print(filtered_steps)


def construct_fileformatter():
    specs = []
    specs.append(conditions.ConditionsFileformatSpec('temp', '{:d}'))
    specs.append(conditions.ConditionsFileformatSpec('bias', '{:.1f}'))

    return conditions.ConditionsFileformatter(specs)


def construct_conditions(args, fileformatter, system_file):
    stack_biases = []
    for stack_mult in args.stack_mults:
        stack_bias = biases.StackingBias(args.stack_ene, stack_mult)
        stack_biases.append(stack_bias)

    conditions_map = {'temp': args.temps,
                      'staple_m': [args.staple_m],
                      'bias': stack_biases}

    return conditions.AllSimConditions(conditions_map, fileformatter, system_file)


def create_input_filepathbase(args):
    return '{}/{}'.format(args.input_dir, args.filebase)


def create_output_filepathbase(args):
    return '{}/{}'.format(args.output_dir, args.filebase)


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
        '--reps',
        nargs='+',
        type=int,
        help='Reps (leave empty for all available)')
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
    parser.add_argument(
        '--tags',
        type=str,
        nargs='+',
        help='List of order parameter tags')
    parser.add_argument(
        '--values',
        type=int,
        nargs='+',
        help='List of order parameter values to filter for')

    return parser.parse_args()


if __name__ == '__main__':
    main()
