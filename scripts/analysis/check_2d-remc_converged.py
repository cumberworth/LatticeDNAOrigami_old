#!/usr/bin/env python

"""Check if REMC simulations converged

Convergence criteria is based on number of steps following all replicas having
sampled a fully stacked state.
"""

import argparse

from origamipy import biases
from origamipy import conditions
from origamipy import decorrelate
from origamipy import outputs

import numpy as np


def main():
    args = parse_args()
    system_file = files.JSONStructInpFile(args.system_filename)
    fileformatter = construct_fileformatter()
    all_conditions = construct_conditions(args, fileformatter, system_file)
    inp_filebase = create_input_filepathbase(args)
    sim_collections = outputs.create_sim_collections(inp_filebase,
                                                     all_conditions, args.reps)
    reps = len(sim_collections[0]._reps)
    reps_converged = [False for i in range(reps)]
    for rep in range(reps):
        for sim_collection in sim_collections:
            ops = sim_collection.get_reps_data('ops')[rep]
            stacked_pairs = ops['numstackedpairs']
            fully_stacked = np.where(
                stacked_pairs == args.fully_stacked_pairs)[0]
            if len(fully_stacked) == 0:
                continue

            steps_since_fully_stacked = len(stacked_pairs) - fully_stacked[0]
            if steps_since_fully_stacked > args.prod_steps:
                reps_converged[rep] = True
                break

    if np.all(reps_converged):
        print(1)
    else:
        print(0)


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
        'staple_m',
        type=float,
        help='Staple molarity (mol/V)')
    parser.add_argument(
        'stack_ene',
        type=float,
        help='Stacking energy (kb K)')
    parser.add_argument(
        'fully_stacked_pairs',
        type=int,
        help='Number of stacked pairs in fully stacked state')
    parser.add_argument(
        'prod_steps',
        type=int,
        help='Minimum number of steps to run after fully stacked states sampled')
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
