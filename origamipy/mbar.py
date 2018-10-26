"""Running MBAR."""

import collections
import itertools
import math

import numpy as np
from pymbar import timeseries
from pymbar import mbar

from origamipy import biases
from origamipy import datatypes


NUM_STAPLES_TAG = 'numstaples'


def calc_reduced_potentials(enes, ops, conditions):
    """Reduced potentials as defined in shirts2008."""
    rstaple_u = calc_reduced_staple_u(conditions)
    bias_collection = conditions.bias(ops)
    num_staples = ops[NUM_STAPLES_TAG]
    e = (enes.enthalpies + enes.stacking_energies + bias_collection)
    renes = e/conditions.temp
    return renes - enes.enthalpies + rstaple_u*num_staples


def calc_reduced_staple_u(conditions):
    """Calculate reduced staple chemical potential"""
    rstaple_u = math.log(conditions.staple_m)
    return rstaple_u


DecorrelationResults = collections.namedtuple('DecorrelationResults', [
                                              'start_i',
                                              'statistical_inefficiency',
                                              'Neff'])


SimConditions = collections.namedtuple('SimConditions', [
                                       'temp',
                                       'staple_m',
                                       'bias',
                                       'fileformat'])


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


class AllSimConditions:
    """All combinations of given simulation conditions."""
    def __init__(self, condition_map, fileformatter):
        self._conditions_map = condition_map
        self._fileformatter = fileformatter
        self._combos = None
        self._cur_conditions = None
        self._cur_fileformat = None
        self._cur_total_bias = None

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

        self._cur_conditions = dict(zip(self._conditions_map.keys(), combo))
        self._construct_total_bias()
        self._cur_fileformat = self._fileformatter(self._cur_conditions)
        return self.current_conditions

    def _construct_total_bias(self):
        bs = [v for k, v in self._cur_conditions.items() if 'bias' in k]
        self._cur_total_bias = biases.TotalBias(bs)
        return self.current_conditions

    @property
    def temp(self):
        return self._cur_conditions['temp']

    @property
    def staple_m(self):
        return self._cur_conditions['staple_m']

    @property
    def bias(self):
        return self._cur_total_bias

    @property
    def fileformat(self):
        return self._cur_fileformat

    @property
    def current_conditions(self):
        c = SimConditions(self.temp, self.staple_m, self.bias, self.fileformat)
        return c


class SimCollection:
    _datatypes = ['enes', 'ops', 'staples', 'staplestates']
    """Output data for all runs of each replica of a simulation."""
    def __init__(self, filebase, conditions):
        self._filebase = filebase
        self._conditions = conditions
        self._datatype_to_reps = {tag: [] for tag in self._datatypes}
        self._reps_masks = []
        self._datatype_to_decorrelated = {}
        self._decorrelation_performed = False
        self._num_reps = 0
        self._reps_num_steps = []
        self._num_decorrelated_steps = 0

        self._load_reps_data()

    def _load_reps_data(self):
        reps_remain = True
        rep = -1
        while reps_remain:
            rep += 1
            reps_remain = self._load_runs_data(rep)
            self._get_num_steps()

        self._num_reps = rep

    def _load_runs_data(self, rep):
        datatype_to_runs = {key: [] for key in self._datatype_to_reps.keys()}
        runs_remain = True
        run = 0
        while runs_remain:
            template = '{}_run-{}_rep-{}-{}'
            filebase = template.format(self._filebase, run, rep,
                                       self._conditions.fileformat)
            temp = self._conditions.temp
            try:
                datatype_to_runs = self._load_run_data(filebase,
                                                       datatype_to_runs, temp)
            except IOError:
                runs_remain = False
                break

            run += 1

        if run == 0 and not runs_remain:
            return False

        else:
            for key in datatype_to_runs.keys():
                c = datatypes.OutputData.concatenate(datatype_to_runs[key])
                self._datatype_to_reps[key].append(c)

            return True

    def _get_num_steps(self):
        steps = len(self._datatype_to_reps['enes'][-1]['step'])
        self._reps_num_steps.append(steps)

    def _load_run_data(self, filebase, datatype_to_runs, temp):
        enes = datatypes.Energies.from_file(filebase, temp)
        datatype_to_runs['enes'].append(enes)
        ops = datatypes.OrderParams.from_file(filebase)
        datatype_to_runs['ops'].append(ops)
        staples = datatypes.NumStaplesOfType.from_file(filebase)
        datatype_to_runs['staples'].append(staples)
        staplestates = datatypes.StapleTypeStates.from_file(filebase)
        datatype_to_runs['staplestates'].append(staplestates)

        return datatype_to_runs

    def perform_decorrelation(self):
        for rep in range(self._num_reps):
            self._construct_decorrelation_mask(rep)

        self._apply_masks_and_concatenate()
        self._num_decorrelated_steps = sum([len(m) for m in self._reps_masks])
        self._decorrelation_performed = True

    def _construct_decorrelation_mask(self, rep):
        enes = self.reps_energies[rep]
        ops = self.reps_order_params[rep]
        rpots = calc_reduced_potentials(enes, ops, self._conditions)
        results = DecorrelationResults(*timeseries.detectEquilibration(rpots))
        template = '{:<8} {:<8} {:<3} {:<4.1f} {:<.1f}'
        print(template.format(self._conditions.fileformat,
              self._reps_num_steps[rep], results.start_i,
              results.statistical_inefficiency, results.Neff))
        indices = (timeseries.subsampleCorrelatedData(rpots[results.start_i:],
                   g=results.statistical_inefficiency))
        self._reps_masks.append([i + results.start_i for i in indices])

    def _apply_masks_and_concatenate(self):
        for datatype, reps in self._datatype_to_reps.items():
            r = datatypes.OutputData.concatenate_with_masks(reps,
                                                            self._reps_masks)
            self._datatype_to_decorrelated[datatype] = r

    @property
    def reps_energies(self):
        return self._datatype_to_reps['enes']

    @property
    def reps_order_params(self):
        return self._datatype_to_reps['ops']

    @property
    def reps_staples(self):
        return self._datatype_to_reps['staples']

    @property
    def reps_staplestates(self):
        return self._datatype_to_reps['staplestates']

    @property
    def decorrelated_energies(self):
        return self._get_decorrelated_data('enes')

    @property
    def decorrelated_order_params(self):
        return self._get_decorrelated_data('ops')

    @property
    def decorrelated_staples(self):
        return self._get_decorrelated_data('staples')

    @property
    def decorrelated_staplestates(self):
        return self._get_decorrelated_data('staplestates')

    @property
    def num_decorrelated_configs(self):
        return self._num_decorrelated_steps

    def _get_decorrelated_data(self, tag):
        if self._decorrelation_performed:
            return self._datatype_to_decorrelated[tag]
        else:
            print("Decorrleation has not been performed")
            raise Exception


class MultiStateSimCollection:
    """Output data from a parallel simulation with multiple states."""
    def __init__(self, filepathbase, all_conditions):
        self._filepathbase = filepathbase
        self._all_conditions = all_conditions
        self._sim_collections = []
        self._decorrelated_enes = []
        self._decorrelated_ops = []
        self._decorrelated_staples = []
        self._decorrelated_staplestates = []

        self._create_sim_collections()

    def _create_sim_collections(self):
        for conditions in self._all_conditions:
            sim_collection = SimCollection(self._filepathbase, conditions)
            self._sim_collections.append(sim_collection)

    def perform_decorrelation(self):
        print('State,   configs, t0, g,   Neff')
        for sim_collection in self._sim_collections:
            sim_collection.perform_decorrelation()

        self._collect_decorrelated_outputs()

    def _collect_decorrelated_outputs(self):
        enes = []
        ops = []
        staples = []
        staplestates = []
        for sim in self._sim_collections:
            enes.append(sim.decorrelated_energies)
            ops.append(sim.decorrelated_order_params)
            staples.append(sim.decorrelated_staples)
            staplestates.append(sim.decorrelated_staplestates)

        self._decor_enes = datatypes.OutputData.concatenate(enes)
        self._decor_ops = datatypes.OutputData.concatenate(ops)
        self._decor_staples = datatypes.OutputData.concatenate(staples)
        s = datatypes.OutputData.concatenate(staplestates)
        self._decor_staplestates = s

    def perform_mbar(self):
        rpots_matrix = self._calc_decorrelated_rpots_for_all_conditions()
        num_configs_per_conditions = self._get_num_configs_per_conditions()
        self._mbar = mbar.MBAR(rpots_matrix, num_configs_per_conditions)

    def _calc_decorrelated_rpots_for_all_conditions(self):
        rpots = []
        for conditions in self._all_conditions:
            rpots.append(calc_reduced_potentials(self._decor_enes,
                         self._decor_ops, conditions))

        return np.array(rpots)

    def _get_num_configs_per_conditions(self):
        num_configs = []
        for sim in self._sim_collections:
            num_configs.append(sim.num_decorrelated_configs)

        return num_configs

#    def calculate_expectations(self):
#        for tag in tags:
#            aves, varis = self._mbar.computeExpectations(
#                    uncorrelated_ops[tag], target_uncorrelated_rpots)
#            stds = np.sqrt(varis)
#
#        return aves, stds
