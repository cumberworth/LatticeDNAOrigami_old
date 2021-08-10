"""Methods for decorrelating simulation results."""

import copy
import math

import numpy as np
from pymbar import timeseries

from origamipy import datatypes
from origamipy import utility


NUM_STAPLES_TAG = 'numstaples'


class DecorrelatedOutputs:
    _datatypes = ['enes', 'ops', 'staples', 'staplestates']
    _trjtypes = ['trj', 'vcf', 'ores', 'states']
#    _trjtypes = ['trj', 'vcf', 'states']

    def __init__(self, sim_collections, all_conditions):
        self.all_conditions = all_conditions
        self._sim_collections = sim_collections
        self._decor_masks = []
        self._num_decorrelated_steps = 0
        self._datatype_to_decors = {}
        self._trjtype_to_decors = {}

    def get_concatenated_datatype(self, tag):
        concat = []
        for data in self._datatype_to_decors[tag]:
            concat.append(datatypes.OutputData.concatenate(data))

        return datatypes.OutputData.concatenate(concat)

    def get_num_steps_per_condition(self):
        steps = []
        for rep_to_data in self._datatype_to_decors['enes']:
            steps.append(sum([s.steps for s in rep_to_data]))

        return steps

    def get_concatenated_series(self, tag):
        for reps_data in self._datatype_to_decors.values():
            if tag in reps_data[0][0].tags:
                concat = []
                for data in reps_data:
                    reps = []
                    for series in data:
                        reps.append(series[tag])

                    concat.append(np.concatenate(reps))

                return np.concatenate(concat)

        else:
            raise Exception

    @property
    def all_series_tags(self):
        tags = []
        for decors in self._datatype_to_decors.values():
            datatype = decors[0][0]
            for tag in datatype.tags:
                if tag == 'step':
                    continue

                tags.append(tag)

        return tags

    def perform_decorrelation(self, skip):
        print('Performing decorrelations')
        print('State,   configs, t0, g,   Neff')
        for sim_collection in self._sim_collections:
            self._decor_masks.append([])
            for i, rep in enumerate(sim_collection._reps):
                mask = self._construct_decorrelation_mask(sim_collection, rep,
                        skip)
                self._decor_masks[-1].append(mask)

    def _construct_decorrelation_mask(self, sim_collection, rep, skip):
        enes = sim_collection.reps_energies[rep]
        ops = sim_collection.reps_order_params[rep]
        num_staples = sim_collection.reps_staples[rep]
        steps = enes.steps
        rpots = utility.calc_reduced_potentials(enes, ops, num_staples,
                                                sim_collection.conditions)
        start_i, g, Neff = timeseries.detectEquilibration(rpots, nskip=skip)
        template = '{:<8} {:<8} {:<3} {:<4.1f} {:<.1f}'
        print(template.format(sim_collection.conditions.fileformat, steps,
                start_i, g, Neff))
        indices = (timeseries.subsampleCorrelatedData(rpots[start_i:], g=skip*g))
        return [i + start_i for i in indices]

    def read_decors_from_files(self, data_only=False):
        for datatype in self._datatypes:
            self._datatype_to_decors[datatype] = []
            for sim_collection in self._sim_collections:
                reps_series = sim_collection.get_decor_reps_data(datatype)
                self._datatype_to_decors[datatype].append(reps_series)

        if not data_only:
            for trjtype in self._trjtypes:
                self._trjtype_to_decors[trjtype] = []
                for sim_collection in self._sim_collections:
                    reps_series = sim_collection.get_decor_reps_trj(trjtype)
                    self._trjtype_to_decors[trjtype].append(reps_series)

    def apply_masks(self):
        # The mask numbering is different than the rep number
        for datatype in self._datatypes:
            self._datatype_to_decors[datatype] = []
            for i, sim_collection in enumerate(self._sim_collections):
                self._datatype_to_decors[datatype].append([])
                reps_to_data = sim_collection.get_reps_data(datatype)
                for j, rep in enumerate(sim_collection._reps):
                    data = reps_to_data[rep]
                    data.apply_mask(self._decor_masks[i][j])
                    self._datatype_to_decors[datatype][i].append(data)

        for trjtype in self._trjtypes:
            self._trjtype_to_decors[trjtype] = []
            for i, sim_collection in enumerate(self._sim_collections):
                self._trjtype_to_decors[trjtype].append([])
                reps_to_trjs = sim_collection.get_reps_trj(trjtype)
                for j, rep in enumerate(sim_collection._reps):
                    trjs = reps_to_trjs[rep]
                    filebase = sim_collection.decor_filebase_template.format(
                            sim_collection.filebase, rep,
                            sim_collection.conditions.fileformat)
                    filename = '{}.{}'.format(filebase, trjtype)
                    decor_trj = self._apply_mask_to_trjs(self._decor_masks[i][j],
                            trjs, filename)
                    self._trjtype_to_decors[trjtype][i].append(decor_trj)

    def _apply_mask_to_trjs(self, mask, trjs, filename):
        out_file = open(filename, 'w')
        step_i = 0
        mask_i = 0
        for trj in trjs:
            for step in trj:
                step_included = step_i == mask[mask_i]
                if step_included:
                    out_file.write(step)
                    mask_i += 1
                    if mask_i == len(mask):
                        return

                step_i += 1
            trj.close()

        out_file.close()

    def write_decors_to_files(self):
        for datatype in self._datatypes:
            for i, sim_collection in enumerate(self._sim_collections):
                for j, rep in enumerate(sim_collection._reps):
                    filebase = sim_collection.decor_filebase_template.format(
                            sim_collection.filebase, rep,
                            sim_collection.conditions.fileformat)
                    if datatype == 'enes':
                        self._datatype_to_decors[datatype][i][j].to_file(filebase,
                                float(sim_collection.conditions.temp))
                    else:
                        self._datatype_to_decors[datatype][i][j].to_file(filebase)

    def filter_collections(self, filter_tag, value):
        filtered_count = 0
        for i, sim_collection in enumerate(self._sim_collections):
            for j, rep in enumerate(sim_collection._reps):

                # Create mask
                selected_op = self._datatype_to_decors['ops'][i][j][filter_tag]
                mask = selected_op == value
                filtered_count += mask.sum()

                # Apply mask
                for datatype in self._datatypes:
                    data = self._datatype_to_decors[datatype][i][j]._data
                    reduced_data = []
                    for series in data:
                        reduced_data.append(series[mask])

                    reduced_data = np.array(reduced_data)
                    self._datatype_to_decors[datatype][i][j]._data = reduced_data

        return filtered_count


class SimpleDecorrelatedOutputs:
    _datatypes = ['enes', 'ops', 'staples', 'staplestates']
    _trjtypes = ['trj', 'vcf', 'ores', 'states']

    def __init__(self, sim_collections, all_conditions):
        self.all_conditions = all_conditions
        self._sim_collections = sim_collections
        self._decor_masks = []
        self._num_decorrelated_steps = 0
        self._datatype_to_decors = {}
        self._trjtype_to_decors = {}

    def get_concatenated_datatype(self, tag):
        return datatypes.OutputData.concatenate(self._datatype_to_decors[tag])

    def get_num_steps_per_condition(self):
        steps = []
        for data in self._datatype_to_decors['enes']:
            steps.append(data.steps)

        return steps

    def get_concatenated_series(self, tag):
        for data in self._datatype_to_decors.values():
            if tag in data[0].tags:
                concat = []
                for series in data:
                    concat.append(series[tag])

                return np.concatenate(concat)

        else:
            raise Exception

    @property
    def all_series_tags(self):
        tags = []
        for decors in self._datatype_to_decors.values():
            datatype = decors[0]
            for tag in datatype.tags:
                if tag == 'step':
                    continue

                tags.append(tag)

        return tags

    def perform_decorrelation(self, skip):
        print('Performing decorrelations')
        print('State,   configs, t0, g,   Neff')
        for sim_collection in self._sim_collections:
            mask = self._construct_decorrelation_mask(sim_collection, skip)
            self._decor_masks.append(mask)

    def _construct_decorrelation_mask(self, sim_collection, skip):
        enes = sim_collection.get_data('enes')
        ops = sim_collection.get_data('ops')
        num_staples = sim_collection.get_data('staples')
        steps = enes.steps
        rpots = utility.calc_reduced_potentials(enes, ops, num_staples,
                                                sim_collection.conditions)
        start_i, g, Neff = timeseries.detectEquilibration(rpots, nskip=skip)
        template = '{:<8} {:<8} {:<3} {:<4.1f} {:<.1f}'
        print(template.format(sim_collection.conditions.fileformat, steps,
                start_i, g, Neff))
        indices = (timeseries.subsampleCorrelatedData(rpots[start_i:], g=skip*g))
        return [i + start_i for i in indices]

    def read_decors_from_files(self, data_only=False):
        for datatype in self._datatypes:
            self._datatype_to_decors[datatype] = []
            for sim_collection in self._sim_collections:
                series = sim_collection.get_data(datatype)
                self._datatype_to_decors[datatype].append(series)

        if not data_only:
            for trjtype in self._trjtypes:
                self._trjtype_to_decors[trjtype] = []
                for sim_collection in self._sim_collections:
                    series = sim_collection.get_trj(trjtype)
                    self._trjtype_to_decors[trjtype].append(series)

    def apply_masks(self):
        # The mask numbering is different than the rep number
        for datatype in self._datatypes:
            self._datatype_to_decors[datatype] = []
            for i, sim_collection in enumerate(self._sim_collections):
                data = sim_collection.get_data(datatype)
                data.apply_mask(self._decor_masks[i])
                self._datatype_to_decors[datatype].append(data)

        for trjtype in self._trjtypes:
            self._trjtype_to_decors[trjtype] = []
            for i, sim_collection in enumerate(self._sim_collections):
                trj = sim_collection.get_trj(trjtype)
                filebase = '{}_decor'.format(sim_collection.filebase)
                filename = '{}.{}'.format(filebase, trjtype)
                decor_trj = self._apply_mask_to_trj(self._decor_masks[i],
                        trj, filename)
                self._trjtype_to_decors[trjtype].append(decor_trj)

    def _apply_mask_to_trj(self, mask, trj, filename):
        out_file = open(filename, 'w')
        step_i = 0
        mask_i = 0
        for step in trj:
            step_included = step_i == mask[mask_i]
            if step_included:
                out_file.write(step)
                mask_i += 1
                if mask_i == len(mask):
                    return

            step_i += 1
        trj.close()

        out_file.close()

    def write_decors_to_files(self):
        for datatype in self._datatypes:
            for i, sim_collection in enumerate(self._sim_collections):
                filebase = '{}_decor'.format(sim_collection.filebase)
                if datatype == 'enes':
                    self._datatype_to_decors[datatype][i].to_file(filebase,
                            float(sim_collection.conditions.temp))
                else:
                    self._datatype_to_decors[datatype][i].to_file(filebase)

    def filter_collections(self, filter_tag, value):
        filtered_count = 0
        for i, sim_collection in enumerate(self._sim_collections):

            # Create mask
            selected_op = self._datatype_to_decors['ops'][i][filter_tag]
            mask = selected_op == value
            filtered_count += mask.sum()

            # Apply mask
            for datatype in self._datatypes:
                data = self._datatype_to_decors[datatype][i]._data
                reduced_data = []
                for series in data:
                    reduced_data.append(series[mask])

                reduced_data = np.array(reduced_data)
                self._datatype_to_decors[datatype][i]._data = reduced_data

        return filtered_count
