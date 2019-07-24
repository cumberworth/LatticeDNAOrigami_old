"""Bias functions."""

import json

import numpy as np


STACK_TAG = 'numstackedpairs'


class NoBias:
    def __call__(self, *args):
        return 0

    @property
    def fileformat_value(self):
        return 0


class StackingBias:
    def __init__(self, stack_energy, stack_mult):
        self._stack_energy = stack_energy
        self._stack_mult = stack_mult
        self._complementary_stack_mult = 1 - float(stack_mult)

    def __call__(self, order_params):
        total_stack_energy = order_params[STACK_TAG]*self._stack_energy
        return -total_stack_energy*self._complementary_stack_mult

    @property
    def fileformat_value(self):
        return self._stack_mult


class GridBias:
    """Grid bias with linear step well outside grid."""
    def __init__(self, tags, window, min_outside_bias, slope, inp_filebase):
        self._tags = tags
        self._window = window
        self._min_outside_bias = min_outside_bias
        self._slope = slope

        # Create window file postfix
        self._postfix = '_win'
        for win_min in window[0]:
            self._postfix += '-' + str(win_min)

        self._postfix += '-'
        for win_max in window[1]:
            self._postfix += '-' + str(win_max)
        
        # Read biases from file
        filename = '{}{}.biases'.format(inp_filebase, self._postfix)
        grid_biases = json.load(open(filename))
        self._grid_biases = {}
        for entry in grid_biases['biases']:
            point = tuple(entry['point'])
            bias = entry['bias']
            self._grid_biases[point] = bias

        self._postfix += '_iter-prod'

    def __call__(self, order_params):
        biases = []
        for step in range(order_params.steps):
            point = tuple(order_params[tag][step] for tag in self._tags)
            bias = 0

            # Grid bias
            if point in self._grid_biases.keys():
                bias += self._grid_biases[point]

            # Linear step bias
            for i, param in enumerate(point):
                min_param = self._window[0][i]
                max_param = self._window[1][i]
                if param < min_param:
                    bias += self._slope * (min_param - param - 1)
                    bias += self._min_outside_bias;
                elif param > max_param:
                    bias = self._slope * (param - max_param - 1)
                    bias += self._min_outside_bias;

            biases.append(bias)

        return np.array(biases);

    @property
    def fileformat_value(self):
        return self._postfix


class TotalBias:
    def __init__(self, biases):
        self._biases = biases

    def __call__(self, order_params):
        return sum([bias(order_params) for bias in self._biases])
