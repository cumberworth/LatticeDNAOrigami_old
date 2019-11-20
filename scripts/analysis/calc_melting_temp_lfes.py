#!/usr/bin/env python

"""Estimate melting temperature and calculate LFEs.

It is assumed there is a barrier between the unassembled and assembled states.
The melting temperature here is defined as that at which the local minima on
either side of the barrier are equal. For the this temperature, the LFEs are
calculate across the selected order parameter.
"""

import argparse
import sys

import numpy as np
from scipy.signal import argrelextrema
from scipy import interpolate
from scipy.optimize import minimize

from origamipy import biases
from origamipy import conditions
from origamipy import files
from origamipy import outputs
from origamipy import decorrelate
from origamipy import mbar_wrapper
from origamipy import utility


def main():
    args = parse_args()
    system_file = files.JSONStructInpFile(args.system_filename)
    fileformatter = construct_fileformatter()
    all_conditions = construct_conditions(args, fileformatter, system_file)
    staple_lengths = all_conditions._staple_lengths
    inp_filebase = create_input_filepathbase(args)
    sim_collections = outputs.create_sim_collections(inp_filebase,
                                                     all_conditions, args.reps)
    decor_outs = decorrelate.DecorrelatedOutputs(
        sim_collections, all_conditions)
    decor_outs.read_decors_from_files()

    mbarw = mbar_wrapper.MBARWrapper(decor_outs)
    mbarw.perform_mbar()

    out_filebase = create_output_filepathbase(args)

    values = decor_outs.get_concatenated_series(args.tag)
    decor_enes = decor_outs.get_concatenated_datatype('enes')
    decor_ops = decor_outs.get_concatenated_datatype('ops')
    halfway_temp = estimate_halfway_temp(mbarw, values, all_conditions,
                                         args.assembled_op)
    if args.guess_temp is not None:
        halfway_temp = args.guess_temp

    print('Halfway temperature: {:.3f} K'.format(
        np.around(halfway_temp, decimals=3)))
    bins = list(set(values))
    bins.sort()
    value_to_bin = {value: i for i, value in enumerate(bins)}
    bin_index_series = [value_to_bin[i] for i in values]
    bin_index_series = np.array(bin_index_series)
    conds = conditions.SimConditions({'temp': halfway_temp, 'staple_m': args.staple_m,
                                      'bias': biases.NoBias()}, fileformatter, staple_lengths)
    melting_temp = minimize(squared_barrier_diff, halfway_temp,
                            args=(mbarw, values, bins, bin_index_series, decor_enes, decor_ops,
                                  conds)).x[0]
    lfes, lfe_stds = calc_lfes(mbarw, conds, bins, bin_index_series, decor_enes,
                               decor_ops)
    barrier_height = np.around(calc_forward_barrier_height(lfes), decimals=3)
    barrier_i = find_barrier(lfes)
    melting_temp = '{:.3f}'.format(np.around(melting_temp, decimals=3))

    print('Estimated melting temperature: {} K'.format(melting_temp))
    print('Barrier height: {:.3f} kT'.format(barrier_height))
    print('Barrier peak: {:.3f}'.format(bins[barrier_i]))

    header = np.array(['ops', melting_temp])
    lfes_filebase = '{}_{}-lfes-melting'.format(out_filebase, args.tag)

    lfes_file = files.TagOutFile('{}.aves'.format(lfes_filebase))
    lfes = np.concatenate([[bins], [lfes]]).T
    lfes_file.write(header, lfes)

    stds_file = files.TagOutFile('{}.stds'.format(lfes_filebase))
    lfe_stds = np.concatenate([[bins], [lfe_stds]]).T
    stds_file.write(header, lfe_stds)

    # Calculate expectations along the selected OP
    mbarws = []
    all_decor_outs = []
    sampled_ops = []
    for i in range(1, args.assembled_op + 1):
        sim_collections = outputs.create_sim_collections(inp_filebase,
                                                         all_conditions, args.reps)
        decor_outs = decorrelate.DecorrelatedOutputs(
            sim_collections, all_conditions)
        decor_outs.read_decors_from_files(data_only=True)
        filtered_count = decor_outs.filter_collections(args.tag, i)
        if filtered_count == 0:
            continue

        sampled_ops.append(i)
        all_decor_outs.append(decor_outs)
        mbarw = mbar_wrapper.MBARWrapper(decor_outs)
        mbarw.perform_mbar()
        mbarws.append(mbarw)

    out_filebase = create_output_filepathbase(args)

    # Calculate expectations across selected order parameter
    all_tags = []
    for i in range(1, args.staple_types + 1):
        all_tags.append('staples{}'.format(i))
        all_tags.append('staplestates{}'.format(i))

    for i in range(args.scaffold_domains):
        all_tags.append('domainstate{}'.format(i))

    aves, stds = calc_reduced_expectations(
        conds, mbarws, all_decor_outs, all_tags)

    aves = np.concatenate([[sampled_ops], np.array(aves).T])
    aves_file = files.TagOutFile('{}-{}.aves'.format(out_filebase, args.tag))
    aves_file.write([args.tag] + all_tags, aves.T)

    stds = np.concatenate([[sampled_ops], np.array(stds).T])
    stds_file = files.TagOutFile('{}-{}.stds'.format(out_filebase, args.tag))
    stds_file.write([args.tag] + all_tags, stds.T)


def calc_reduced_expectations(conds, mbarws, all_decor_outs, tags):
    all_aves = []
    all_stds = []
    for mbarw, decor_outs in zip(mbarws, all_decor_outs):
        aves = []
        stds = []
        for tag in tags:
            series = decor_outs.get_concatenated_series(tag)
            ave, std = mbarw.calc_expectation(series, conds)
            aves.append(ave)
            stds.append(std)

        all_aves.append(aves)
        all_stds.append(stds)

    return all_aves, all_stds


def calc_forward_barrier_height(lfes):
    barrier_i = find_barrier(lfes)
    minima = find_minima(lfes, barrier_i)

    return lfes[barrier_i] - minima[0]


def estimate_halfway_temp(mbarw, values, all_conditions, max_op):
    temps = [float(conds.temp) for conds in all_conditions]
    aves, stds = mbarw._calc_expectations(values)
    interpolated_temp = interpolate.interp1d(aves, temps, kind='linear')

    return interpolated_temp(max_op/2)


def squared_barrier_diff(temp, mbarw, values, bins, bin_index_series,
                         decor_enes, decor_ops, conds):
    conds._conditions['temp'] = temp
    lfes, lfes_stds = calc_lfes(mbarw, conds, bins, bin_index_series, decor_enes,
                                decor_ops)
    barrier_i = find_barrier(lfes)
    minima = find_minima(lfes, barrier_i)

    return (minima[0] - minima[1])**2


def calc_lfes(mbarw, conds, bins, bin_index_series, decor_enes, decor_ops):
    rpots = utility.calc_reduced_potentials(decor_enes, decor_ops,
                                            conds)

    return mbarw._mbar.computePMF(
        rpots, bin_index_series, len(bins))


def find_barrier(lfes):
    # Find largest maximum
    maxima_i = argrelextrema(lfes, np.greater)[0]
    if len(maxima_i) == 0:
        print('No barrier detected')
        sys.exit()

    maxima = lfes[maxima_i]
    maximum_i = maxima_i[maxima.argmax()]

    return maximum_i


def find_minima(lfes, maximum_i):
    lower_lfes = lfes[:maximum_i]
    upper_lfes = lfes[maximum_i:]

    return (lower_lfes.min(), upper_lfes.min())


def parse_tag_pairs(tag_pairs):
    return [tuple(tag_pair.split(',')) for tag_pair in tag_pairs]


def construct_conditions(args, fileformatter, system_file):
    conditions_map = {'temp': args.temps,
                      'staple_m': [args.staple_m],
                      'bias': [biases.NoBias()]}

    return conditions.AllSimConditions(conditions_map, fileformatter, system_file)


def construct_fileformatter():
    specs = [conditions.ConditionsFileformatSpec('temp', '{}')]
    return conditions.ConditionsFileformatter(specs)


def create_input_filepathbase(args):
    return '{}/{}'.format(args.input_dir, args.filebase)


def create_output_filepathbase(args):
    return '{}/{}'.format(args.output_dir, args.filebase)


def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('tag',
                        type=str,
                        help='Order parameter tag')
    parser.add_argument('assembled_op',
                        type=int,
                        help='Value of order parameter in assembled state')
    parser.add_argument('staple_types',
                        type=int,
                        help='Number of staple types')
    parser.add_argument('scaffold_domains',
                        type=int,
                        help='Number of scaffold domains')
    parser.add_argument(
        '--guess_temp',
        type=float,
        help='Temperature (K)')
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

    return parser.parse_args()


if __name__ == '__main__':
    main()
