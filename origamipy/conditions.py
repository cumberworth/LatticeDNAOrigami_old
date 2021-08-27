"""Simulation conditions."""

import collections
import itertools
import json

import numpy as np

from origamipy import biases
from origamipy import us_process


ConditionsFileformatSpec = collections.namedtuple(
    'ConditionFileformatSpec', ['condition', 'spec'])


class ConditionsFileformatter:
    def __init__(self, spec):
        self._spec = spec

    def __call__(self, cur_conditions):

        fileformat_elements = []
        for condition, spec in self._spec:
            condition_value = cur_conditions[condition]
            if 'bias' in condition:
                condition_value = condition_value.fileformat_value

            fileformat_elements.append(spec.format(condition_value))

        return '-'.join(fileformat_elements)


class SimConditions:
    def __init__(self, conditions, fileformat, staple_lengths):
        self._conditions = conditions
        self._fileformat = fileformat
        self._total_bias = None

        self._construct_total_bias()
        self._u_extra_states_term = (2*staple_lengths - 1)*np.log(6)

    def _construct_total_bias(self):
        bs = [v for k, v in self._conditions.items() if 'bias' in k]
        self._total_bias = biases.TotalBias(bs)

    def __getitem__(self, key):
        return self._conditions[key]

    @property
    def temp(self):
        return self['temp']

    @property
    def staple_m(self):
        return self['staple_m']

    @property
    def reduced_staple_us(self):
        return np.log(self['staple_m']) - self._u_extra_states_term

    @property
    def total_bias(self):
        return self._total_bias

    @property
    def fileformat(self):
        return self._fileformat

    @property
    def condition_tags(self):
        return list(sorted(self._conditions.keys()))

    @property
    def characteristic_values(self):
        char_values = []
        for key, value in sorted(self._conditions.items()):
            if 'bias' in key:
                value = value.fileformat_value

            char_values.append(value)

        return char_values

    @property
    def condition_to_characteristic_value(self):
        condition_to_char_value = {}
        for key, value in sorted(self._conditions.items()):
            if 'bias' in key:
                value = value.fileformat_value

            condition_to_char_value[key] = value

        return condition_to_char_value


class AllSimConditions:
    """All combinations of given simulation conditions."""

    def __init__(self, conditions_keys, conditions_values, fileformatter,
                 staple_lengths):
        self._conditions_keys = conditions_keys
        self._conditions_values = conditions_values
        self._fileformatter = fileformatter
        self._staple_lengths = staple_lengths
        self._conditions = None
        self._fileformat = None
        self._total_bias = None
        self._combos = None

        self._reset_combo_iterator()

    def _reset_combo_iterator(self):
        combos = []
        for vs in self._conditions_values:
            combos.append(itertools.product(*vs))

        self._combos = itertools.chain(*combos)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            combo = next(self._combos)
        except StopIteration:
            self._reset_combo_iterator()
            raise

        self._conditions = dict(zip(self._conditions_keys, combo))
        self._fileformat = self._fileformatter(self._conditions)

        return SimConditions(self._conditions, self._fileformat, self._staple_lengths)

    @property
    def condition_tags(self):
        return list(sorted(self._conditions.keys()))

    @property
    def conditions_to_characteristic_values(self):
        condition_values = []
        for conditions in self:
            values = conditions.characteristic_value
            condition_values.append(
                [v for k, v in sorted(char_values.items())])

        return condition_values


# I don't like how many arguemnts this requires, but I don't know how else to
# organize it
def construct_mwus_conditions(
        windows_filename, bias_functions_filename, reps, start_run, temp, itr,
        staple_m, fileformatter, inp_filebase, staple_lengths, concatenate):

    bias_tags, windows = us_process.read_windows_file(windows_filename)
    bias_functions = json.load(open(bias_functions_filename))
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
    for rep in range(reps):
        grid_biases = []
        for window in windows:
            filebase = '{}_run-{}_rep-{}'.format(
                inp_filebase, start_run, rep)
            grid_biases.append(
                biases.GridBias(
                    op_tags, window, min_outside_bias, slope, temp,
                    filebase, itr))

        conditions_valuesl.append([[temp], [staple_m], grid_biases])

    if concatenate:
        return AllSimConditions(
            conditions_keys, conditions_valuesl, fileformatter,
            staple_lengths)
    else:
        reps_conditions = []
        for conditions_values in conditions_valuesl:
            reps_conditions.append(AllSimConditions(
                conditions_keys, [conditions_values], fileformatter,
                staple_lengths))

        return reps_conditions
