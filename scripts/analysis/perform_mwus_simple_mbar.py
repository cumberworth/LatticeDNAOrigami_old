#!/usr/bin/env python

"""Carry out standard MBAR analysis on MWUS simulation output."""

import argparse
import json

import numpy as np

from origamipy import biases
from origamipy import conditions
from origamipy import decorrelate
from origamipy import files
from origamipy import mbar_wrapper
from origamipy import outputs
from origamipy import us_process
from origamipy import utility


def main():
    args = parse_args()
    system_file = files.JSONStructInpFile(args.system_filename)
    inp_filebase = create_input_filepathbase(args)
    fileformatter = construct_fileformatter()
    all_conditions = construct_conditions(
        args, fileformatter, inp_filebase, system_file)
    staple_lengths = all_conditions._staple_lengths
    sim_collections = create_simplesim_collections(args, inp_filebase,
                                                   all_conditions)
    decor_outs = decorrelate.SimpleDecorrelatedOutputs(
        sim_collections, all_conditions)
    decor_outs.read_decors_from_files()

    mbarw = mbar_wrapper.MBARWrapper(decor_outs)
    mbarw.perform_mbar()

    out_filebase = create_output_filepathbase(args)
    conds = conditions.SimConditions({'temp': args.temp, 'staple_m': args.staple_m,
                                      'bias': biases.NoBias()}, fileformatter, staple_lengths)
    all_tags = decor_outs.all_conditions.condition_tags
    aves = []
    stds = []
    for tag in decor_outs.all_series_tags:
        all_tags.append(tag)
        series = decor_outs.get_concatenated_series(tag)
        ave, std = mbarw.calc_expectation(series, conds)
        aves.append(ave)
        stds.append(std)

        # Hack calculate LFEs
        values = decor_outs.get_concatenated_series(tag)
        decor_enes = decor_outs.get_concatenated_datatype('enes')
        decor_ops = decor_outs.get_concatenated_datatype('ops')
        decor_staples = decor_outs.get_concatenated_datatype('staples')
        bins = list(set(values))
        bins.sort()
        value_to_bin = {value: i for i, value in enumerate(bins)}
        bin_index_series = [value_to_bin[i] for i in values]
        bin_index_series = np.array(bin_index_series)
        rpots = utility.calc_reduced_potentials(decor_enes, decor_ops,
                                                decor_staples, conds)
        lfes, lfe_stds = mbarw._mbar.computePMF(
            rpots, bin_index_series, len(bins))

        # Hack write LFEs to file
        header = np.array(['ops', args.temp])
        lfes_filebase = '{}_{}-lfes-melting'.format(out_filebase, tag)
        lfes_file = files.TagOutFile('{}.aves'.format(lfes_filebase))
        lfes = np.concatenate([[bins], [lfes]]).T
        lfes_file.write(header, lfes)
        stds_file = files.TagOutFile('{}.stds'.format(lfes_filebase))
        lfe_stds = np.concatenate([[bins], [lfe_stds]]).T
        stds_file.write(header, lfe_stds)

    # Hack to write expectations to file
    aves_file = files.TagOutFile('{}.aves'.format(out_filebase))
    cond_char_values = conds.condition_to_characteristic_value
    cond_values = [v for k, v in sorted(cond_char_values.items())]
    aves_file.write(all_tags, [np.concatenate([cond_values, np.array(aves)])])
    stds_file = files.TagOutFile('{}.stds'.format(out_filebase))
    stds_file.write(all_tags, [np.concatenate([cond_values, np.array(stds)])])


def parse_tag_pairs(tag_pairs):
    return [tuple(tag_pair.split(',')) for tag_pair in tag_pairs]


def construct_fileformatter():
    specs = [conditions.ConditionsFileformatSpec('bias', '{}')]
    return conditions.ConditionsFileformatter(specs)


def construct_conditions(args, fileformatter, inp_filebase, system_file):
    bias_tags, windows = us_process.read_windows_file(args.windows_filename)
    bias_functions = json.load(open(args.bias_functions_filename))
    op_tags = us_process.get_op_tags_from_bias_functions(
        bias_functions, bias_tags)

    # Linear square well functions are all the same
    for bias_function in bias_functions['origami']['bias_functions']:
        if bias_function['type'] == 'LinearStepWell':
            slope = bias_function['slope']
            min_outside_bias = bias_function['min_bias']

    grid_biases = []
    for window in windows:
        for rep in range(args.reps):
            filebase = '{}_run-{}_rep-{}'.format(inp_filebase, args.run, rep)
            grid_biases.append(biases.GridBias(op_tags, window,
                                               min_outside_bias, slope,
                                               args.temp, filebase, args.itr))

    conditions_map = {'temp': [args.temp],
                      'staple_m': [args.staple_m],
                      'bias': grid_biases}

    # either get rid of this too or make a list of filebases for creating sim collections
    return conditions.AllSimConditions(conditions_map, fileformatter, system_file)


def create_simplesim_collections(args, inp_filebase, all_conditions):
    sim_collections = []
    rep = 0
    for conditions in all_conditions:
        filebase = '{}_run-{}_rep-{}{}_decor'.format(inp_filebase, args.run, rep,
                                                     conditions.fileformat)
        sim_collection = outputs.SimpleSimCollection(
            filebase, conditions, args.reps)
        sim_collections.append(sim_collection)
        rep += 1
        rep %= args.reps

    return sim_collections


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
        'windows_filename',
        type=str,
        help='Windows filename')
    parser.add_argument(
        'bias_functions_filename',
        type=str,
        help='Bias functions filename')
    parser.add_argument(
        'temp',
        type=float,
        help='Temperature (K)')
    parser.add_argument(
        'staple_m',
        type=float,
        help='Staple molarity (mol/V)')
    parser.add_argument(
        'stack_ene',
        type=float,
        help='Stacking energy (kb K)')
    parser.add_argument(
        'reps',
        type=int,
        help='Number of reps')
    parser.add_argument(
        'run',
        type=int,
        help='Number of reps')
    parser.add_argument(
        'itr',
        type=int,
        help='US iteration')
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
