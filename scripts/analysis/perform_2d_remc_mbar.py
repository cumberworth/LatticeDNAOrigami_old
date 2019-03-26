#!/usr/bin/env python

"""Carry out standard MBAR analysis on 2D REMC simulation output.

The exchange variables are assumed to be temperature and stacking multiplier,
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
#    mbarw.calc_all_expectations(out_filebase)
    reduced_conditions = construct_variable_temp_conditions(
        args, 0, fileformatter)
    mbarw.calc_1d_lfes(reduced_conditions, out_filebase + '-0.0')
    mbarw.calc_specified_2d_lfes(parse_tag_pairs(args.tag_pairs),
            reduced_conditions, out_filebase + '-0.0')
    #mbarw.calc_staplestates_tag_lfes(
    #        'numstackedpairs', reduced_conditions, out_filebase)


def construct_conditions(args, fileformatter):
    stack_biases = []
    for stack_mult in args.stack_mults:
        stack_bias = biases.StackingBias(args.stack_ene, stack_mult)
        stack_biases.append(stack_bias)

    conditions_map = {'temp': args.temps,
                      'staple_m': [args.staple_m],
                      'bias': stack_biases}

    return conditions.AllSimConditions(conditions_map, fileformatter)


def construct_fileformatter():
    specs = []
    specs.append(conditions.ConditionsFileformatSpec('temp', '{:d}'))
    specs.append(conditions.ConditionsFileformatSpec('bias', '{:.1f}'))

    return conditions.ConditionsFileformatter(specs)


def create_input_filepathbase(args):
    return '{}/{}'.format(args.input_dir, args.filebase)


def create_output_filepathbase(args):
    return '{}/{}'.format(args.output_dir, args.filebase)


def construct_variable_temp_conditions(args, stack_mult, fileformatter):
    stack_bias = biases.StackingBias(args.stack_ene, stack_mult)
    conditions_map = {'temp': args.temps,
                      'staple_m': [args.staple_m],
                      'bias': [stack_bias]}

    return conditions.AllSimConditions(conditions_map, fileformatter)


def parse_tag_pairs(tag_pairs):
    return [tuple(tag_pair.split(',')) for tag_pair in tag_pairs]


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
        'skip',
        type=int,
        help='Number of steps to skip')
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
        '--tag_pairs',
        nargs='+',
        type=str,
        help='Tags to calculate 2D pmf for (comma delim)')

    return parser.parse_args()


if __name__ == '__main__':
    main()
