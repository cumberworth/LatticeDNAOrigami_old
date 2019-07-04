"""Simulation conditions."""

import collections
import itertools

from origamipy import biases


ConditionsFileformatSpec = collections.namedtuple('ConditionFileformatSpec', [
                                                 'condition',
                                                 'spec'])


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
    def __init__(self, conditions, fileformat):
        self._conditions = conditions
        self._fileformat = fileformat
        self._total_bias = None

        self._construct_total_bias()

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
    def total_bias(self):
        return self._total_bias

    @property
    def fileformat(self):
        return self._fileformat

    @property
    def condition_to_characteristic_value(self):
        condition_to_char_value = {}
        for key, value in self._conditions.items():
            if 'bias' in key:
                value = value.fileformat_value

            condition_to_char_value[key] = float(value)

        return condition_to_char_value


class AllSimConditions:
    """All combinations of given simulation conditions."""
    def __init__(self, condition_map, fileformatter):
        self._conditions_map = condition_map
        self._fileformatter = fileformatter
        self._conditions = None
        self._fileformat = None
        self._total_bias = None
        self._combos = None

        self._reset_combo_iterator()

    def _reset_combo_iterator(self):
        self._combos = itertools.product(*self._conditions_map.values())

    def __iter__(self):
        return self

    def __next__(self):
        try:
            combo = next(self._combos)
        except StopIteration:
            self._reset_combo_iterator()
            raise

        self._conditions = dict(zip(self._conditions_map.keys(), combo))
        self._fileformat = self._fileformatter(self._conditions)

        return SimConditions(self._conditions, self._fileformat)

    @property
    def condition_tags(self):
        return list(self._conditions.keys())

    @property
    def condition_to_characteristic_values(self):
        condition_values = []
        for conditions in self:
            char_values = conditions.condition_to_characteristic_value
            condition_values.append([v for v in char_values.values()])

        return condition_values
