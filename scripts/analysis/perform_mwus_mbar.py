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
    inp_filebase = '{}/{}'.format(args.input_dir, args.filebase)
    fileformatter = construct_fileformatter()
    reps_all_conditions = construct_conditions(
        args, fileformatter, inp_filebase, system_file, False)
    sim_collections = []
    for rep in range(args.reps):
        rep_sim_collections = outputs.create_sim_collections(
            inp_filebase,
            reps_all_conditions[rep],
            rep,
            args.start_run,
            args.end_run)
        sim_collections.append(rep_sim_collections)

    all_conditions = construct_conditions(
        args, fileformatter, inp_filebase, system_file, True)
    decor_outs = decorrelate.DecorrelatedOutputs(
        sim_collections, all_conditions=all_conditions,
        rep_conditions_equal=False)
    decor_outs.read_decors_from_files()

    mbarw = mbar_wrapper.MBARWrapper(decor_outs)
    mbarw.perform_mbar()

    out_filebase = '{}/{}_run-{}-{}_iter-{}'.format(
        args.output_dir,
        args.filebase,
        args.start_run,
        args.end_run,
        args.itr)

    staple_lengths = all_conditions._staple_lengths
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


def construct_conditions(args, fileformatter, inp_filebase, system_file,
                         concatenate):
    bias_tags, windows = us_process.read_windows_file(args.windows_filename)
    bias_functions = json.load(open(args.bias_functions_filename))
    op_tags = us_process.get_op_tags_from_bias_functions(
        bias_functions, bias_tags)

    # Linear square well functions are all the same
    for bias_function in bias_functions['origami']['bias_functions']:
        if bias_function['type'] == 'LinearStepWell':
            slope = bias_function['slope']
            min_outside_bias = bias_function['min_bias']
            break

    conditions_keys = ['temp', 'staple_m', 'bias']
    conditions_valuesl = []
    for rep in range(args.reps):
        grid_biases = []
        for window in windows:
            filebase = '{}_run-{}_rep-{}'.format(
                inp_filebase, args.start_run, rep)
            grid_biases.append(
                biases.GridBias(
                    op_tags, window, min_outside_bias, slope, args.temp,
                    filebase, args.itr))

        conditions_valuesl.append([[args.temp], [args.staple_m], grid_biases])

    if concatenate:
        return conditions.AllSimConditions(
            conditions_keys, conditions_valuesl, fileformatter,
            system_file)
    else:
        reps_conditions = []
        for conditions_values in conditions_valuesl:
            reps_conditions.append(conditions.AllSimConditions(
                conditions_keys, [conditions_values], fileformatter,
                system_file))

        return reps_conditions


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
        'start_run',
        type=int,
        help='Run to concatenate from')
    parser.add_argument(
        'end_run',
        type=int,
        help='Run to concatenate to')
    parser.add_argument(
        'itr',
        type=int,
        help='US iteration')
    parser.add_argument(
        '--tags',
        nargs='+',
        type=str,
        help='Order parameter tags')
    parser.add_argument(
        '--tag_pairs',
        nargs='+',
        type=str,
        help='Tags to calculate 2D pmf for (comma delim)')

    return parser.parse_args()


if __name__ == '__main__':
    main()
