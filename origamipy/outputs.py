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
        out_file = files.TagOutFile('{}.aves'.format(filebase))
        out_file.write(tags, np.concatenate([all_conds, np.array(means)],
                                            axis=1))


class SimCollection:
    """Output data for all runs of each replica of a simulation."""

    filebase_template = '{}_run-{}_rep-{}-{}'
    decor_filebase_template = '{}_rep-{}-{}_decor'

    def __init__(self, filebase, conditions, reps):
        self.conditions = conditions
        self.filebase = filebase
        self._datatype_to_reps = {}
        self._trjtype_to_reps = {}
        self._num_reps = 0

        # This is ugly
        self._reps = reps
        self._find_num_reps()

    # Maybe get rid of properties and off access by tag and rep
    @property
    def reps_energies(self):
        return self.get_reps_data('enes')

    @property
    def reps_order_params(self):
        return self.get_reps_data('ops')

    @property
    def reps_staples(self):
        return self.get_reps_data('staples')

    @property
    def reps_staplestates(self):
        return self.get_reps_data('staplestates')

    @property
    def decorrelated_energies(self):
        return self.get_decor_reps_data('enes')

    @property
    def decorrelated_order_params(self):
        return self.get_decor_reps_data('ops')

    @property
    def decorrelated_staples(self):
        return self.get_decor_reps_data('staples')

    @property
    def decorrelated_staplestates(self):
        return self.get_decor_reps_data('staplestates')

    def get_reps_data(self, tag):
        if tag not in self._datatype_to_reps.keys():
            self._datatype_to_reps[tag] = []
            for rep in range(max(self._reps) + 1):
                # This is ugly
                if rep not in self._reps:
                    self._datatype_to_reps[tag].append(0)
                    continue

                self._load_runs_data(rep, tag)

        return self._datatype_to_reps[tag]

    def get_decor_reps_data(self, tag):
        tag = 'decor_{}'.format(tag)
        if tag not in self._datatype_to_reps.keys():
            self._datatype_to_reps[tag] = []
            for rep in range(max(self._reps) + 1):
                # This is ugly
                if rep not in self._reps:
                    self._datatype_to_reps[tag].append(0)
                    continue

                self._load_concat_data(rep, tag)

        return self._datatype_to_reps[tag]

    def get_reps_trj(self, tag):
        all_trjs = []
        if tag not in self._trjtype_to_reps.keys():
            self._trjtype_to_reps[tag] = []
            for rep in range(max(self._reps) + 1):
                if rep not in self._reps:
                    self._trjtype_to_reps[tag].append([])
                    continue

                all_trjs.append(self._load_runs_trj(rep, tag))

            self._trjtype_to_reps[tag] = all_trjs

        return self._trjtype_to_reps[tag]

    def _load_runs_trj(self, rep, tag):
        runs_remain = True
        run = 0
        trjs = []
        while runs_remain:
            filebase = self.filebase_template.format(self.filebase, run, rep,
                    self.conditions.fileformat)
            filename = '{}.{}'.format(filebase, tag)
            try:
                if tag in ['trj', 'vcf']:
                    trj = files.UnparsedMultiLineStepInpFile(filename, 0)
                    trjs.append(trj)
                elif tag in ['ores', 'states']:
                    trj = files.UnparsedSingleLineStepInpFile(filename, 0)
                    trjs.append(trj)
                else:
                    NotImplementedError

            except IOError:
                runs_remain = False
                break

            run += 1

        return trjs

    def get_decor_reps_trj(self, tag):
        tag = 'decor_{}'.format(tag)
        all_trjs = []
        if tag not in self._trjtype_to_reps.keys():
            self._trjtype_to_reps[tag] = []
            for rep in range(max(self._reps) + 1):
                if rep not in self._reps:
                    self._trjtype_to_reps[tag].append([])
                    continue

                filebase = self.filebase_template.format(self.filebase, run,
                        rep, self.conditions.fileformat)
                try:
                    if tag in ['trj', 'vcf']:
                        trj = files.UnparsedMultiLineStepInpFile(filename, 0)
                    else:
                        raise NotImplementedError
                except IOError:
                    print('Decorrelation not performed')
                    raise Exception

                all_trjs.append(trj)

            self._trjtype_to_reps[tag] = all_trjs

        return self._trjtype_to_reps[tag]

    def _find_num_reps(self):
        reps_remain = True
        rep = -1
        while reps_remain:
            rep += 1
            t = '{}_run-0_rep-{}-{}.ops'
            fname = t.format(self.filebase, rep, self.conditions.fileformat)
            reps_remain = os.path.isfile(fname)

        self._num_reps = rep
        if self._reps == None:
            self._reps = list(range(rep))

    def _load_runs_data(self, rep, tag):
        runs_remain = True
        run = 0
        all_series = []
        while runs_remain:
            filebase = self.filebase_template.format(self.filebase, run, rep,
                    self.conditions.fileformat)
            try:
                all_series.append(self._load_data_from_file(filebase, tag))
            except IOError:
                runs_remain = False
                break

            run += 1

        series = datatypes.OutputData.concatenate(all_series)
        self._datatype_to_reps[tag].append(series)

    def _load_concat_data(self, rep, tag):
        filebase = self.decor_filebase_template.format(self.filebase, rep,
                self.conditions.fileformat)
        try:
            series = self._load_data_from_file(filebase, tag)
        except IOError:
            print('Decorrelation not performed')
            raise Exception

        self._datatype_to_reps[tag].append(series)

    def _load_data_from_file(self, filebase, tag):
        if 'enes' in tag:
            series = datatypes.Energies.from_file(filebase, self.conditions.temp)
        if 'ops' in tag:
            series = datatypes.OrderParams.from_file(filebase)
        if 'staples' in tag:
            series = datatypes.NumStaplesOfType.from_file(filebase)
        if 'staplestates' in tag:
            series = datatypes.StapleTypeStates.from_file(filebase)

        return series


def create_sim_collections(filebase, all_conditions, reps):
    sim_collections = []
    for conditions in all_conditions:
        sim_collection = SimCollection(filebase, conditions, reps)
        sim_collections.append(sim_collection)

    return sim_collections
