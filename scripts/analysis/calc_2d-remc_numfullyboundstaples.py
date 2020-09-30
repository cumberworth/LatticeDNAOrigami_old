#!/usr/bin/env python

"""Calculate total number of fully bound staples for a simulation set"""


import argparse
import os.path

import numpy as np

from origamipy import biases
from origamipy import conditions
from origamipy import datatypes
from origamipy import files
from origamipy import outputs
from origamipy import decorrelate
from origamipy import mbar_wrapper


def main():
    args = parse_args()
    system_file = files.JSONStructInpFile(args.system_filename)
    fileformatter = construct_fileformatter()
    all_conditions = construct_conditions(args, fileformatter, system_file)
    inp_filebase = create_input_filepathbase(args)
    sim_collections = outputs.create_sim_collections(inp_filebase,
                                                     all_conditions, args.reps)

    tag = 'numfullyboundstaples'
    for sim_collection in sim_collections:
        staple_states = sim_collection.get_reps_data(
            'staplestates', concatenate=False)
        for rep in sim_collection._reps:
            runs = len(staple_states[rep])
            for run in range(runs):
                back_ops_filebase = sim_collection.get_filebase(run, rep)
                back_ops = datatypes.OrderParams.from_file(back_ops_filebase)
                total_staples = staple_states[rep][run]._data[1:, :].sum(
                    axis=0)
                if tag in back_ops.tags:
                    back_ops[tag] = total_staples
                else:
                    back_ops.add_column(tag, total_staples)

                back_ops.to_file(back_ops_filebase)


def construct_conditions(args, fileformatter, system_file):
    stack_biases = []
    for stack_mult in args.stack_mults:
        stack_bias = biases.StackingBias(args.stack_ene, stack_mult)
        stack_biases.append(stack_bias)

    conditions_map = {'temp': args.temps,
                      'staple_m': [args.staple_m],
                      'bias': stack_biases}

    return conditions.AllSimConditions(conditions_map, fileformatter, system_file)


def construct_fileformatter():
    specs = []
    specs.append(conditions.ConditionsFileformatSpec('temp', '{}'))
    specs.append(conditions.ConditionsFileformatSpec('bias', '{}'))

    return conditions.ConditionsFileformatter(specs)


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
