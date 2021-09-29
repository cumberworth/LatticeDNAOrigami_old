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
    staple_lengths = utility.calc_staple_lengths(system_file)
    inp_filebase = f'{args.input_dir}/{args.filebase}'
    fileformatter = construct_fileformatter()
    all_conditions = conditions.construct_remc_conditions(
        args.temps, args.staple_m, fileformatter, staple_lengths)
    sim_collections = []
    for rep in range(args.reps):
        rep_sim_collections = outputs.create_sim_collections(
            inp_filebase, all_conditions, rep)
        sim_collections.append(rep_sim_collections)

    decor_outs = decorrelate.DecorrelatedOutputs(
        sim_collections, all_conditions=all_conditions,
        rep_conditions_equal=True)
    decor_outs.read_decors_from_files()

    mbarw = mbar_wrapper.MBARWrapper(decor_outs)
    mbarw.perform_mbar()

    se_tags = decor_outs.all_series_tags
    out_filebase = '{}/{}_run-{}-{}'.format(
        args.output_dir,
        args.filebase,
        decor_outs.start_run,
        decor_outs.end_run)

    halfway_temp = estimate_halfway_temp(
        mbarw, values, all_conditions, args.assembled_op)
    if args.guess_temp is not None:
        halfway_temp = args.guess_temp

    print('Guess temperature: {:.3f} K'.format(
        np.around(halfway_temp, decimals=3)))
    conds = conditions.SimConditions(
        {'temp': args.guess_temp,
         'staple_m': args.staple_m,
         'bias': biases.NoBias()},
        fileformatter, staple_lengths)
    bias = biases.NoBias()
    melting_temp = est_melting_temp_and_barrier(
        mbarw, fileformatter, staple_lengths, conds, bias, args)
    conds = conditions.SimConditions(
        {'temp': melting_temp,
         'staple_m': args.staple_m,
         'bias': biases.NoBias()},
        fileformatter, staple_lengths)

    lfes_filebase = f'{out_filebase}_lfes-melting'
    mbarw.calc_all_1d_lfes(lfes_filebase, se_tags, [conds])
    mbarw.calc_all_expectations(out_filebase, se_tags, [conds])

    # Calculate expectations along OP slices
    mbarws = []
    all_decor_outs = []
    sampled_ops = []
    for i in range(1, args.assembled_op + 1):
        sim_collections = outputs.create_sim_collections(
            inp_filebase, all_conditions, args.reps)
        decor_outs = decorrelate.DecorrelatedOutputs(
            sim_collections, all_conditions=all_conditions,
            rep_conditions_equal=True)
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

    # Calculate expectations for staple and domain states
    all_tags = []
    for i in range(1, args.staple_types + 1):
        all_tags.append(f'staples{i}')
        all_tags.append(f'staplestates{i}')

    for i in range(args.scaffold_domains):
        all_tags.append(f'domainstate{i}')

    aves, stds = calc_reduced_expectations(
        conds, mbarws, all_decor_outs, all_tags)

    aves = np.concatenate([[sampled_ops], np.array(aves).T])
    aves_file = files.TagOutFile('{out_filebase}-{args.tag}.aves')
    aves_file.write([args.tag] + all_tags, aves.T)

    stds = np.concatenate([[sampled_ops], np.array(stds).T])
    stds_file = files.TagOutFile('{out_filebase}-{args.tag}.stds')
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


def est_melting_temp_and_barrier(
        mbarw, fileformatter, staple_lengths, conds, bias, args):

    try:
        melting_temp = mbarw.estimate_melting_temp(conds, args.temp)
    except:
        melting_temp = mbarw.estimate_melting_temp_endpoints(conds, args.temp)

    conds = conditions.SimConditions(
        {'temp': melting_temp,
         'staple_m': args.staple_m,
         'bias': bias},
        fileformatter, staple_lengths)

    melting_temp_f = '{:.3f}'.format(np.around(melting_temp, decimals=3))
    print(f'Estimated melting temperature: {melting_temp_f} K')
    for se_tag in ['numfullyboundstaples', 'numfulldomains']:
        lfes, stds, bins = mbarw.calc_1d_lfes(se_tag, conds)
        try:
            barrier_i = mbar_wrapper.find_barrier(lfes)
            barrier_height = mbar_wrapper.calc_forward_barrier_height(lfes)
            print()
            print(f'Barrier height, {se_tag}: {barrier_height:.3f} kT')
            print(f'Barrier peak, {se_tag}: {bins[barrier_i]:.3f}')
        except:
            pass

    return melting_temp


def estimate_halfway_temp(mbarw, values, all_conditions, max_op):
    temps = [float(conds.temp) for conds in all_conditions]
    aves, stds = mbarw._calc_expectations(values)
    interpolated_temp = interpolate.interp1d(aves, temps, kind='linear')

    return interpolated_temp(max_op/2)


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
        'tag',
        type=str,
        help='Order parameter tag')
    parser.add_argument(
        'assembled_op',
        type=int,
        help='Value of order parameter in assembled state')
    parser.add_argument(
        'staple_types',
        type=int,
        help='Number of staple types')
    parser.add_argument(
        'scaffold_domains',
        type=int,
        help='Number of scaffold domains')
    parser.add_argument(
        'reps',
        type=int,
        help='Number of reps')
    parser.add_argument(
        '--guess_temp',
        type=float,
        help='Temperature (K)')
    parser.add_argument(
        '--temps',
        nargs='+',
        type=str,
        help='Temperatures')

    return parser.parse_args()


if __name__ == '__main__':
    main()
