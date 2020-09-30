#!/usr/bin/env python

"""Carry out standard MBAR analysis on 2D REMC simulation output.

The exchange variables are assumed to be temperature and stacking multiplier,
in that order.
"""

import argparse

from origamipy import conditions
from origamipy import biases
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
    decor_outs = decorrelate.DecorrelatedOutputs(
        sim_collections, all_conditions)
    decor_outs.read_decors_from_files()

    mbarw = mbar_wrapper.MBARWrapper(decor_outs)
    mbarw.perform_mbar()

    out_filebase = create_output_filepathbase(args)
    mbarw.calc_all_expectations(out_filebase)
    for smult in args.stack_mults:
        reduced_conditions = construct_variable_temp_conditions(
            args, smult, fileformatter, system_file)
        reduced_out_filebase = '{}-{}'.format(out_filebase, smult)
        mbarw.calc_1d_lfes(reduced_conditions, reduced_out_filebase)


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


def construct_variable_temp_conditions(args, stack_mult, fileformatter, system_file):
    stack_bias = biases.StackingBias(args.stack_ene, stack_mult)
    conditions_map = {'temp': args.temps,
                      'staple_m': [args.staple_m],
                      'bias': [stack_bias]}

    return conditions.AllSimConditions(conditions_map, fileformatter, system_file)


def parse_tag_pairs(tag_pairs):
    return [tuple(tag_pair.split(',')) for tag_pair in tag_pairs]


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
    parser.add_argument(
        '--tag_pairs',
        nargs='+',
        type=str,
        help='Tags to calculate 2D pmf for (comma delim)')

    return parser.parse_args()


if __name__ == '__main__':
    main()
