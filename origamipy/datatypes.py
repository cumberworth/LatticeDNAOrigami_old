"""Functions for processing order parameter files"""

import numpy as np
import pandas as pd


class EnumerationWeights:
    def __init__(self, filename):
        self._tags, self._dataframe = self._read_weights_from_file(filename)

    @classmethod
    def _read_weights_from_file(cls, filename):
        data = pd.read_csv(filename, sep='\s|\)\s', header=None, skiprows=1,
                           engine='python')
        data[0] = data[0].str.lstrip('(').astype(int)
        tags = cls._get_tags(filename)
        tags.append('weight')
        data.columns = tags
        tags.pop()
        return tags, data

    @classmethod
    def _get_tags(cls, filename):
        tags = []
        with open(filename) as f:
            tags = f.readline().rstrip().split()
            tags = [tag.rstrip(',') for tag in tags]

        return tags

    def calc_all_1d_means(self):
        means = []
        for tag in self._tags:
            weights = self._marginalize(tag)
            means.append((weights.index*weights).sum())

        return means

    def calc_1d_lfes(self, ops, tag):
        lfes = []
        for op in ops:
            w = self._dataframe[self._dataframe[tag] == op]['weight'].sum()
            lfes.append(-np.log(w))

        lfes = np.array(lfes)
        lfes -= lfes.min()

        return lfes

    def get_op_range(self, tag):
        min_op = self._dataframe[tag].min()
        max_op = self._dataframe[tag].max()

        return list(range(min_op, max_op + 1))

    def _marginalize(self, tag):
        return self._dataframe.groupby(tag)['weight'].sum()

    @property
    def tags(self):
        return self._tags


class OutputData:
    """Base class for output datatypes"""
    @classmethod
    def from_file(cls, filebase):
        filename = cls._create_filename(filebase)
        data = cls._load_file(filename)
        header = cls._get_header(filename)
        tags = cls._get_tags(header, data=data)

        return cls(header, tags, data)

    @classmethod
    def _create_filename(cls, filebase):
        return '{}.{}'.format(filebase, cls._ext)

    @classmethod
    def _load_file(cls, filename):
        data = np.loadtxt(filename, skiprows=cls._header_lines,
                          dtype=cls._dtype, ndmin=2)
        return data.transpose()

    @classmethod
    def _get_header(cls, filename):
        if cls._header_lines == 0:
            return ''

        with open(filename) as f:
            return f.readline()

    @classmethod
    def _get_tags(cls, header, **kwargs):
        tags = header.rstrip().split()
        tags = [tag.rstrip(',') for tag in tags]

        return tags

    @classmethod
    def concatenate(cls, output_data_list):
        data = output_data_list[-1]

        # UGLY HACK TO CUT EXTRA COLUMSN IN OPS
        cols = len(data._tags)
        data_list = [d._data[:cols] for d in output_data_list]
        concatenated_data = np.concatenate(data_list, axis=1)
        return type(data)(data._header, data._tags[:cols], concatenated_data)

    @classmethod
    def concatenate_with_masks(cls, output_data_list, masks):
        data_list = [d._data for d in output_data_list]
        masked_data_list = []
        for data, mask in zip(data_list, masks):
            masked_data_list.append(data.T[mask].T)

        concatenated_data = np.concatenate(masked_data_list, axis=1)
        data = output_data_list[0]
        return type(data)(data._header, data._tags, concatenated_data)

    @property
    def tags(self):
        return self._tags

    @property
    def steps(self):
        return len(self._data[0])

    def __init__(self, header, tags, data):
        self._header = header
        self._tags = tags
        self._data = data
        self._tag_to_index = {tag: i for i, tag in enumerate(tags)}

    def __add__(self, other):
        data = np.concatenate([self._data, other._data], axis=1)
        return type(self)(self._header, self._tags, data)

    def __getitem__(self, tag):
        index = self._tag_to_index[tag]
        return self._data[index]

    def __setitem__(self, tag, value):
        index = self._tag_to_index[tag]
        self._data[index] = value

    def to_file(self, filebase):
        filename = '{}.{}'.format(filebase, self._ext)
        if self._header_lines != 0:
            self._header = ', '.join(self._tags)

        np.savetxt(filename, self._data.T, header=self._header.rstrip('\n'),
                comments='', fmt=self._fmt)

    def apply_mask(self, mask):
        self._data = self._data.T[mask].T

    def add_column(self, tag, data):
        self._tags.append(tag)
        self._tag_to_index = {tag: i for i, tag in enumerate(self._tags)}
        self._data = np.concatenate([self._data, [data]])

class Energies(OutputData):
    _ext = 'ene'
    _header_lines = 1
    _dtype = np.float
    _fmt = '%f'

    @classmethod
    def from_file(cls, filebase, temp):
        self = super().from_file(filebase)
        self._multiply_energy_by_temp(temp)
        return self

    def __init__(self, header, tags, data):
        super().__init__(header, tags, data)


    def _multiply_energy_by_temp(self, temp):

        # Energy are written in units of be k_b / K
        self['tenergy'] = self['tenergy']*temp
        self['henthalpy'] = self['henthalpy']*temp
        self['stacking'] = self['stacking']*temp


    def to_file(self, filebase, temp):
        self._divide_energy_by_temp(temp)
        super().to_file(filebase)

    def _divide_energy_by_temp(self, temp):

        # Energy are written in units of be k_b / K
        self['tenergy'] = self['tenergy']/temp
        self['henthalpy'] = self['henthalpy']/temp
        self['stacking'] = self['stacking']/temp

    @property
    def total_energies(self):
        return self['tenergy']

    @property
    def enthalpies(self):
        return self['henthalpy']

    @property
    def entropies(self):
        return self['hentropy']

    @property
    def stacking_energies(self):
        return self['stacking']

    @property
    def bias_energies(self):
        return self['bias']

class OrderParams(OutputData):
    _ext = 'ops'
    _header_lines = 1
    _dtype = np.int
    _fmt = '%d'

    @classmethod
    def _get_tags(cls, header, **kwargs):
        tags = super()._get_tags(header)
        if 'step' not in tags:
            tags.insert(0, 'step')

        return tags


class Times(OutputData):
    _ext = 'times'
    _header_lines = 1
    _dtype = np.float
    _fmt = '%f'


class NumStaplesOfType(OutputData):
    _ext = 'staples'
    _header_lines = 0
    _dtype = np.int
    _fmt = '%d'

    @classmethod
    def _get_tags(cls, *args, data=None):
        tags = ['step']
        for i in range(1, data.shape[0]):
            tags.append('staples{}'.format(i))

        return tags


class StapleTypeStates(OutputData):
    _ext = 'staplestates'
    _header_lines = 0
    _dtype = np.int
    _fmt = '%d'

    @classmethod
    def _get_tags(cls, *args, data=None):
        tags = ['step']
        for i in range(1, data.shape[0]):
            tags.append('staplestates{}'.format(i))

        return tags
