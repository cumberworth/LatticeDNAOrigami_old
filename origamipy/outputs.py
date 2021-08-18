"""Collections of simulation and enumeration outputs."""

import os.path

import numpy as np

from origamipy import datatypes
from origamipy import files


class EnumCollection:
    def __init__(self, filebase, all_conditions):
        self._filebase = filebase
        self._all_conditions = all_conditions
        self._enum_weights = []

        self._load_data()

    def _load_data(self):
        for conditions in self._all_conditions:
            t = '{}-{}.weights'
            filename = t.format(self._filebase, conditions.fileformat)
            self._enum_weights.append(datatypes.EnumerationWeights(filename))

    def calc_all_1d_means(self, filebase):
        means = []
        for weights in self._enum_weights:
            means.append(weights.calc_all_1d_means())

        all_conds = self._all_conditions.condition_to_characteristic_values
        tags = self._all_conditions.condition_tags + self._enum_weights[0].tags
        out_file = files.TagOutFile(f'{filebase}.aves')
        out_file.write(tags, np.concatenate([all_conds, np.array(means)],
                                            axis=1))

    def calc_all_1d_lfes(self, filebase):
        for tag in self._enum_weights[0]._tags:
            bins = self._enum_weights[0].get_op_range(tag)
            lfes = []
            for weights in self._enum_weights:
                lfes.append(weights.calc_1d_lfes(bins, tag))

            all_conds = self._all_conditions.condition_to_characteristic_values
            all_tags = self._all_conditions.condition_tags
            temp_i = all_tags.index('temp')
            temps = [c[temp_i] for c in all_conds]
            header = np.concatenate([['ops'], temps])
            bins = np.array(bins).reshape(len(bins), 1)
            data = np.concatenate([bins, np.array(lfes).T], axis=1)
            lfes_filebase = f'{filebase}_{tag}-lfes'
            lfes_file = files.TagOutFile(f'{lfes_filebase}.aves')
            lfes_file.write(header, data)


def create_sim_collections(filebase, all_conditions, rep, start_run=0, end_run=-1):
    sim_collections = []
    for conditions in all_conditions:
        sim_collection = SimCollection(
            filebase, conditions, rep, start_run, end_run)
        sim_collections.append(sim_collection)

    return sim_collections


class SimCollection:
    """A single condition of a simulation."""

    filebase_template = '{}_run-{}_rep-{}{}'
    decor_filebase_template = '{}_run-{}-{}_rep-{}{}_decor'

    def __init__(self, filebase, conditions, rep, start_run=0, end_run=-1):
        self.conditions = conditions
        self.filebase = filebase
        self._datatype = {}
        self._trjtype = {}
        self._rep = rep
        self._start_run = start_run
        self._end_run = end_run

    def get_data(self, dt_tag, concatenate=True):
        """Return datatype series.

        If concatenate, will concatenate all runs to one series.
        """
        if dt_tag not in self._datatype.keys():
            self._load_runs_data(dt_tag, concatenate)

        return self._datatype[dt_tag]

    def get_trj(self, trj_tag):
        all_trjs = []
        if trj_tag not in self._trjtype.keys():
            self._load_runs_trj(trj_tag)

        return self._trjtype[trj_tag]

    def get_filebase(self, run):
        return self.filebase_template.format(
            self.filebase, run, self._rep, self.conditions.fileformat)

    def get_decor_data(self, dt_tag):
        """Return decorrelated datatype series.

        The decorrelated data is already concatenated across runs.
        I wonder if this should be here or on the decor class.
        """
        dt_tag = f'decor_{dt_tag}'
        if dt_tag not in self._datatype.keys():
            self._load_concat_data(dt_tag)

        return self._datatype[dt_tag]

    def get_decor_trj(self, dt_tag):
        dt_tag = f'decor_{dt_tag}'
        if dt_tag not in self._trjtype.keys():
            self._load_concat_trj(dt_tag)

        return self._trjtype[dt_tag]

    def _load_runs_data(self, dt_tag, concatenate):
        runs_remain = True
        run = self._start_run
        all_series = []
        while runs_remain and (self._end_run != -1 and run <= self._end_run):
            filebase = self.filebase_template.format(
                self.filebase, run, self._rep, self.conditions.fileformat)
            try:
                all_series.append(self._load_data_from_file(filebase, dt_tag))
            except IOError:
                runs_remain = False
                break

            run += 1

        if self._end_run == -1:
            self._end_run = run - 1

        if concatenate:
            series = datatypes.OutputData.concatenate(all_series)
            self._datatype[dt_tag] = series
        else:
            self._datatype[dt_tag] = all_series

    def _load_data_from_file(self, filebase, dt_tag):
        if 'enes' in dt_tag:
            series = datatypes.Energies.from_file(
                filebase, float(self.conditions.temp))
        if 'ops' in dt_tag:
            series = datatypes.OrderParams.from_file(filebase)
        if 'staples' in dt_tag:
            series = datatypes.NumStaplesOfType.from_file(filebase)
        # staples is in staplestates. good thing this came after
        if 'staplestates' in dt_tag:
            series = datatypes.StapleTypeStates.from_file(filebase)

        return series

    def _load_runs_trj(self, dt_tag):
        runs_remain = True
        run = self._start_run
        trjs = []
        while runs_remain and (self._end_run != -1 and run <= self._end_run):
            filebase = self.filebase_template.format(
                self.filebase, run, self._rep, self.conditions.fileformat)
            filename = f'{filebase}.{dt_tag}'
            try:
                if dt_tag in ['trj', 'vcf']:
                    trj = files.UnparsedMultiLineStepInpFile(filename, 0)
                    trjs.append(trj)
                elif dt_tag in ['ores', 'states']:
                    trj = files.UnparsedSingleLineStepInpFile(filename, 0)
                    trjs.append(trj)
                else:
                    NotImplementedError

            except IOError:
                runs_remain = False
                break

            run += 1

        self._trjtype[dt_tag] = trjs

    def _load_concat_data(self, dt_tag):
        filebase = self.decor_filebase_template.format(
            self.filebase, self._start_run, self._end_run, self._rep,
            self.conditions.fileformat)
        try:
            series = self._load_data_from_file(filebase, dt_tag)
        except IOError:
            print('Decorrelation not performed')
            raise Exception

        self._datatype[dt_tag] = series

    def _load_concat_trj(self, trj_tag):
        filebase = self.decor_filebase_template.format(
            self.filebase, self._start_run, self._end_run, self._rep,
            self.conditions.fileformat)
        try:
            if 'trj' in trj_tag:
                filename = f'{filebase}.trj'
                trj = files.UnparsedMultiLineStepInpFile(filename, 0)
            elif 'vcf' in trj_tag:
                filename = f'{filebase}.vcf'
                trj = files.UnparsedMultiLineStepInpFile(filename, 0)
            elif 'ores' in trj_tag:
                filename = f'{filebase}.ores'
                trj = files.UnparsedSingleLineStepInpFile(filename, 0)
            elif 'states' in trj_tag:
                filename = f'{filebase}.states'
                trj = files.UnparsedSingleLineStepInpFile(filename, 0)
            else:
                raise NotImplementedError
        except IOError:
            print('Decorrelation not performed')
            raise Exception

        self._trjtype[trj_tag] = trj


class SimpleSimCollection:
    """Output data for all runs of each replica of a simulation."""

    def __init__(self, filebase, conditions):
        self.conditions = conditions
        self.filebase = filebase
        self._datatype = {}
        self._trjtype = {}

    def get_data(self, tag, concatenate=True):
        if tag not in self._datatype.keys():
            self._datatype[tag] = self._load_data(tag)

        return self._datatype[tag]

    def _load_data(self, dt_tag):
        if 'enes' in dt_tag:
            series = datatypes.Energies.from_file(
                self.filebase, float(self.conditions.temp))
        if 'ops' in dt_tag:
            series = datatypes.OrderParams.from_file(self.filebase)
        if 'staples' in dt_tag:
            series = datatypes.NumStaplesOfType.from_file(self.filebase)
        if 'staplestates' in dt_tag:
            series = datatypes.StapleTypeStates.from_file(self.filebase)

        return series

    def get_trj(self, trj_tag):
        if trj_tag not in self._trjtype.keys():
            self._trjtype[trj_tag] = self._load_trj(trj_tag)

        return self._trjtype[tag]

    def _load_trj(self, trj_tag):
        filename = f'{self.filebase}.{tag}'
        if trj_tag in ['trj', 'vcf']:
            trj = files.UnparsedMultiLineStepInpFile(filename, 0)
        elif tag in ['ores', 'states']:
            trj = files.UnparsedSingleLineStepInpFile(filename, 0)
        else:
            NotImplementedError

        return trj
