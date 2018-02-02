"""Functions for preparing and processing US simulations"""

import copy
import json
from operator import itemgetter
import random

import numpy as np

from origamipy.op_process import *

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


def read_win_energies(win_filebases):
    """Read in origami energies for given windows (without bias)"""
    win_enes = []
    for filebase in win_filebases:
        filename = filebase + '_iter-prod.ene'
        enes = np.loadtxt(filename, skiprows=1)
        win_enes.append(enes[:, 1])

    return win_enes


def read_win_energies_from_log(win_filebases):
    """Read in origami energies for given windows from log file (without bias)"""
    win_enes = []
    for filebase in win_filebases:
        filename = filebase + '_iter-prod.out'
        with open(filename) as inp:
            lines = inp.readlines()

        enes =[]
        for line in lines:
            words = line.split()
            if len(words) == 0:
                pass
            else:
                if words[0] == 'System':
                    ene = float(words[-1])
                    enes.append(ene)

        enes = np.array(enes)
        win_enes.append(enes)

    return win_enes


def read_win_order_params(win_filebases, tags):
    """Read in order parameters for given windows"""
    tags.append('numstaples')
    #win_ops = {tag: [] for tag in tags}
    win_ops = []
    for filebase in win_filebases:
        filename = filebase + '_iter-prod.ops'
        ops = read_ops_from_file(filename, tags, 0)
        win_ops.append(ops)

        #for tag in tags:
        #    win_ops[tag].append(ops[tag])

    return win_ops


def read_win_grid_biases(wins, win_filebases):
    """Read grid biases for given windows
    
    Return list of dictionaries indexed by grid point tuples
    """
    win_biases = []
    for filebase in win_filebases:
        filename = filebase + '.biases'
        biases = json.load(open(filename))
        bias_dic = {}
        for entry in biases['biases']:
            point = tuple(entry['point'])
            bias = entry['bias']
            bias_dic[point] = bias

        win_biases.append(bias_dic)

    return win_biases


def get_all_points(win, point, points, comp):
    """
    NOT REALLY SURE
    
    """
    if len(point) == len(win[0]):
        points.append(tuple(point))
    else:
        for i in range(win[0][comp], win[1][comp] + 1):
            point.append(i)
            comp += 1
            points = get_all_points(win, point, points, comp)
            point.pop()
            comp -= 1


    return points


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
