"""Simulation conditions."""

import collections
import itertools

import numpy as np

from origamipy import biases


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
                 system_file):
        self._conditions_keys = conditions_keys
        self._conditions_values = conditions_values
        self._fileformatter = fileformatter
        self._system_file = system_file
        self._conditions = None
        self._fileformat = None
        self._total_bias = None
        self._combos = None

        staple_lengths = [len(ident) for ident in system_file.identities[1:]]
        self._staple_lengths = np.array(staple_lengths)
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
    def condition_to_characteristic_values(self):
        condition_values = []
        for conditions in self:
            char_values = conditions.condition_to_characteristic_value
            condition_values.append(
                [v for k, v in sorted(char_values.items())])

        return condition_values
