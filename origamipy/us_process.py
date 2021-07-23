"""Functions for preparing and processing US simulations"""

import copy
import json
from operator import itemgetter
import random

import numpy as np


def create_win_filename(win, filebase, ext):
    """Create filename for given window."""
    postfix = '_win'
    for win_min in win[0]:
        postfix += '-' + str(win_min)

    postfix += '-'
    for win_max in win[1]:
        postfix += '-' + str(win_max)

    filename = filebase + postfix + ext

    return filename


def create_window_filebases(wins, filebase):
    """Create list of filebases for each window.

    Assumes MWUS simulation conventions for output filenames
    """
    win_filebases = []
    for win in wins:
        postfix = '_win'
        for win_min in win[0]:
            postfix += '-' + str(win_min)

        postfix += '-'
        for win_max in win[1]:
            postfix += '-' + str(win_max)

        win_filebases.append(filebase + postfix)

    return win_filebases


def read_windows_file(filename):
    """Read windows file and return list of tuple of min max tuples"""
    with open(filename) as inp:
        lines = inp.readlines()

    tags = [tag for tag in lines[0].split()]
    wins = []
    for line in lines[1:]:
        mins_raw, maxs_raw = line.split(',')
        mins = tuple(map(int, mins_raw.split()))
        maxs = tuple(map(int, maxs_raw.split()))
        win = (mins, maxs)
        wins.append(win)

    return tags, wins


def get_op_tags_from_bias_functions(bias_functions, bias_tags):
    op_tags = []
    for bias_tag in bias_tags:
        for bias in bias_functions['origami']['bias_functions']:
            if bias_tag == bias['tag']:
                op_tags.append(bias['ops'][0])

    return op_tags


def select_config_by_op_in_win(i, win, ops, op_to_config):
    """Select a configuration with an op in the given window

    Order parameter is selected with uniform probability from range of ops
    allowed in window; configuration is then uniformly selected from
    configurations with the selected order parameters.
    """
    # Select random order parameter in range
    possible_points = get_all_points(win, [], [], 0)

    all_possible_points = copy.deepcopy(possible_points)
    while len(possible_points) != 0:
        point_i = random.randint(0, len(possible_points) - 1)
        point = possible_points[point_i]

        # Select random config with selected op, if available
        if point in op_to_config:
            break
        else:
            possible_points.remove(point)

    else:
        point_i = random.randint(0, len(all_possible_points) - 1)
        point = all_possible_points[point_i]
        print(i)
        points = find_closest_ops(point, op_to_config.keys())
        point_i = random.randint(0, len(points) - 1)
        point = points[point_i]

    possible_configs = op_to_config[point]
    num_possible_config = len(possible_configs)
    sel_config = random.randint(0, num_possible_config - 1)
    config = possible_configs[sel_config]

    return config


def find_closest_ops(point, ops):
    """Find closest ops in given list that is closest to specified value

    Returns a list of all ops that are have the minimum distance found."""
    dist = 0
    points = []
    while len(points) == 0:
        dist += 1
        for op in ops:
            calc_dist = 0
            for i in range(len(op)):
                calc_dist += abs(point[i] - op[i])

            if calc_dist <= dist:
                points.append(op)

    return points
