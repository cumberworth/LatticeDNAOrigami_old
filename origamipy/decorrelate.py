"""Methods for decorrelating simulation results."""

import copy
import math

import numpy as np
from pymbar import timeseries

from origamipy import datatypes
from origamipy import utility


NUM_STAPLES_TAG = 'numstaples'


class DecorrelatedOutputs:
    """Represents decorrelated outputs of a simulation."""
    _datatypes = ['enes', 'ops', 'staples', 'staplestates']
    _trjtypes = ['trj', 'vcf', 'ores', 'states']

    def __init__(self, sim_collections, all_conditions=None,
                 rep_conditions_equal=True):
        self.all_conditions = all_conditions

        self._sim_collections = sim_collections
        self._decor_masks = []
        self._num_decorrelated_steps = 0
        self._datatype_to_decors = {}
        self._trjtype_to_decors = {}
        self._rep_conditions_equal = rep_conditions_equal

    @property
    def all_series_tags(self):
        se_tags = []
        for decors in self._datatype_to_decors.values():
            datatype = decors[0][0]
            for se_tag in datatype.tags:
                if se_tag == 'step':
                    continue

                se_tags.append(se_tag)

        return se_tags

    @property
    def num_steps_per_condition(self):
        steps = []
        for reps_data in self._datatype_to_decors['enes']:
            if self._rep_conditions_equal:
                steps.append(0)
            for data in reps_data:
                if self._rep_conditions_equal:
                    steps[-1] += data.steps
                else:
                    steps.append(data.steps)

        return steps

    def get_concatenated_datatype(self, dt_tag):
        """Concatenate across reps and conditions."""
        concat = []
        if self._rep_conditions_equal:
            for j in range(len(self._datatype_to_decors[0])):
                for rep_data in self._datatype_to_decors[dt_tag]:
                    concat.append(rep_data[j])
        else:
            for rep_data in self._datatype_to_decors[dt_tag]:
                concat.append(datatypes.OutputData.concatenate(rep_data))

        return datatypes.OutputData.concatenate(concat)

    def get_concatenated_series(self, se_tag):
        """Get the raw series data from the datatype."""
        if self._rep_conditions_equal:
            for reps_data in self._datatype_to_decors.values():
                if se_tag in reps_data[0][0]._tags:
                    concat = []
                    for j in range(len(self._datatype_to_decors[0])):
                        for rep_data in reps_data:
                            concat.append(rep_data[j][se_tag])

                    return np.concatenate(concat)
            else:
                raise Exception
        else:
            for reps_data in self._datatype_to_decors.values():
                if se_tag in reps_data[0][0]._tags:
                    concat = []
                    for rep_data in reps_data:
                        conditions = []
                        for data in rep_data:
                            conditions.append(data[se_tag])

                        concat.append(np.concatenate(conditions))

                    return np.concatenate(concat)

            else:
                raise Exception

    def perform_decorrelation(self, skip, detect_equil=False, g=None):
        print('Performing decorrelations')
        print('State'.ljust(20) +
              'configs'.ljust(8) +
              't0'.ljust(8) +
              'g'.ljust(8) +
              'Neff'.ljust(8))
        for rep_sim_collections in self._sim_collections:
            self._decor_masks.append([])
            for sim_collection in rep_sim_collections:
                mask = self._construct_decorrelation_mask(
                    sim_collection, skip, detect_equil, g)
                self._decor_masks[-1].append(mask)

    def read_decors_from_files(self, data_only=False):
        for datatype in self._datatypes:
            self._datatype_to_decors[datatype] = []
            for rep_sim_collections in self._sim_collections:
                self._datatype_to_decors[datatype].append([])
                for sim_collection in rep_sim_collections:
                    data = sim_collection.get_decor_data(datatype)
                    self._datatype_to_decors[datatype][-1].append(data)

        if not data_only:
            for trjtype in self._trjtypes:
                self._trjtype_to_decors[trjtype] = []
                for rep_sim_collections in self._sim_collections:
                    self._trjtype_to_decors[trjtype].append([])
                    for sim_collection in rep_sim_collections:
                        trj = sim_collection.get_decor_trj(trjtype)
                        self._trjtype_to_decors[trjtype][-1].append(trj)

    def apply_masks(self, out_filebase):
        # The mask numbering is different than the rep number
        for datatype in self._datatypes:
            self._datatype_to_decors[datatype] = []
            for i, rep_sim_collections in enumerate(self._sim_collections):
                self._datatype_to_decors[datatype].append([])
                for j, sim_collection in enumerate(rep_sim_collections):
                    data = sim_collection.get_data(datatype)
                    data.apply_mask(self._decor_masks[i][j])
                    self._datatype_to_decors[datatype][-1].append(data)

        # This immediately writes to file
        for trjtype in self._trjtypes:
            self._trjtype_to_decors[trjtype] = []
            for i, rep_sim_collections in enumerate(self._sim_collections):
                self._trjtype_to_decors[trjtype].append([])
                for j, sim_collection in enumerate(rep_sim_collections):
                    trjs = sim_collection.get_trj(trjtype)

                    # Would need to change for when using subset of reps
                    filebase = sim_collection.decor_filebase_template.format(
                        out_filebase, sim_collection._start_run,
                        sim_collection._end_run, i,
                        sim_collection.conditions.fileformat)
                    filename = '{}.{}'.format(filebase, trjtype)
                    decor_trj = self._apply_mask_to_trjs(self._decor_masks[i][j],
                                                         trjs, filename)
                    self._trjtype_to_decors[trjtype][-1].append(decor_trj)

    def write_decors_to_files(self, out_filebase):
        for datatype in self._datatypes:
            for i, rep_sim_collections in enumerate(self._sim_collections):
                for j, sim_collection in enumerate(rep_sim_collections):

                    # Would need to change for when using subset of reps
                    filebase = sim_collection.decor_filebase_template.format(
                        out_filebase, sim_collection._start_run,
                        sim_collection._end_run, i,
                        sim_collection.conditions.fileformat)
                    if datatype == 'enes':
                        self._datatype_to_decors[datatype][i][j].to_file( 
                            filebase, float(sim_collection.conditions.temp))
                    else:
                        self._datatype_to_decors[datatype][i][j].to_file(
                            filebase)

    def filter_collections(self, op_tag, value):
        filtered_count = 0
        for i, rep_sim_collections in enumerate(self._sim_collections):
            for j, sim_collection in enumerate(rep_sim_collections):

                # Create mask
                selected_op = self._datatype_to_decors['ops'][i][j][op_tag]
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

    def _construct_decorrelation_mask(self, sim_collection, skip, detect_equil, g):
        enes = sim_collection.get_data('enes')
        ops = sim_collection.get_data('ops')
        num_staples = sim_collection.get_data('staples')
        steps = enes.steps
        rpots = utility.calc_reduced_potentials(
            enes, ops, num_staples, sim_collection.conditions)
        if g != None:
            start_i = 0
            indices = timeseries.subsampleCorrelatedData(rpots, g=skip*g)
            Neff = len(indices)
        elif detect_equil:
            start_i, g, Neff = timeseries.detectEquilibration(rpots, nskip=skip)
            indices = timeseries.subsampleCorrelatedData(
                rpots[start_i:], g=skip*g)
        else:
            start_i = 0
            g = timeseries.statisticalInefficiency(rpots)
            indices = timeseries.subsampleCorrelatedData(rpots, g=skip*g)
            Neff = len(indices)
            
        template = '{:<20}{:<8}{:<8}{:<8.1f}{:.1f}'
        print(template.format(sim_collection.conditions.fileformat, steps,
                              start_i, g, Neff))
        return [i + start_i for i in indices]

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
