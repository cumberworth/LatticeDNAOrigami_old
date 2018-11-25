"""Collections of simulation and enumeration outputs."""

import collections
import math

import numpy as np
from pymbar import timeseries
from pymbar import mbar

from origamipy import datatypes
from origamipy import io


NUM_STAPLES_TAG = 'numstaples'


def calc_reduced_potentials(enes, ops, conditions):
    """Reduced potentials as defined in shirts2008."""
    rstaple_u = calc_reduced_staple_u(conditions)
    bias_collection = conditions.total_bias(ops)
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
        out_file = io.TagOutFile('{}.aves'.format(filebase))
        out_file.write(tags, np.concatenate([all_conds, np.array(means)],
                       axis=1))


class SimCollection:
    """Output data for all runs of each replica of a simulation."""
    _datatypes = ['enes', 'ops']#, 'staples', 'staplestates']

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
        #staples = datatypes.NumStaplesOfType.from_file(filebase)
        #datatype_to_runs['staples'].append(staples)
        #staplestates = datatypes.StapleTypeStates.from_file(filebase)
        #datatype_to_runs['staplestates'].append(staplestates)

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
        self._sim_collections = {}
        self._decor_enes = []
        self._decor_ops = []
        self._decor_staples = []
        self._decor_staplestates = []
        self._conditions_to_decor_rpots = {}

        self._create_sim_collections()

    def _create_sim_collections(self):
        for conditions in self._all_conditions:
            sim_collection = SimCollection(self._filepathbase, conditions)
            self._sim_collections[conditions] = sim_collection

    def perform_decorrelation(self):
        print('State,   configs, t0, g,   Neff')
        for sim_collection in self._sim_collections.values():
            sim_collection.perform_decorrelation()

        self._collect_decorrelated_outputs()

    def _collect_decorrelated_outputs(self):
        enes = []
        ops = []
        staples = []
        staplestates = []
        for sim in self._sim_collections.values():
            enes.append(sim.decorrelated_energies)
            ops.append(sim.decorrelated_order_params)
            #staples.append(sim.decorrelated_staples)
            #staplestates.append(sim.decorrelated_staplestates)

        self._decor_enes = datatypes.OutputData.concatenate(enes)
        self._decor_ops = datatypes.OutputData.concatenate(ops)
        #self._decor_staples = datatypes.OutputData.concatenate(staples)
        #s = datatypes.OutputData.concatenate(staplestates)
        #self._decor_staplestates = s

    def perform_mbar(self):
        rpots_matrix = self._calc_decorrelated_rpots_for_all_conditions()
        num_configs_per_conditions = self._get_num_configs_per_conditions()
        self._mbar = mbar.MBAR(rpots_matrix, num_configs_per_conditions)

    def _calc_decorrelated_rpots_for_all_conditions(self):
        conditions_rpots = []
        for conditions in self._all_conditions:
            rpots = calc_reduced_potentials(self._decor_enes, self._decor_ops,
                                            conditions)
            conditions_rpots.append(rpots)
            self._conditions_to_decor_rpots[conditions.fileformat] = rpots

        return np.array(conditions_rpots)

    def _get_num_configs_per_conditions(self):
        num_configs = []
        for sim in self._sim_collections.values():
            num_configs.append(sim.num_decorrelated_configs)

        return num_configs

    def calculate_all_expectations(self, filebase):
        dts = [self._decor_ops]#, self._decor_staples, self._decor_staplestates]
        all_tags = self._all_conditions.condition_tags
        all_aves = []
        all_stds = []
        for datatype in dts:
            for tag in datatype.tags:
                if tag == 'step':
                    continue
                aves, stds = self._calc_expectations(datatype[tag])
                all_tags.append(tag)
                all_aves.append(aves)
                all_stds.append(stds)

        all_conds = self._all_conditions.condition_to_characteristic_values
        aves_file = io.TagOutFile('{}.aves'.format(filebase))
        aves_file.write(all_tags, np.concatenate([all_conds,
                        np.array(all_aves).T], axis=1))
        stds_file = io.TagOutFile('{}.stds'.format(filebase))
        stds_file.write(all_tags, np.concatenate([all_conds,
                        np.array(all_stds).T], axis=1))

    def _calc_expectations(self, values):
        aves = []
        stds = []
        for conditions in self._all_conditions:
            # It would be more efficient to collect all timeseries of the same
            # conditions and run this once
            rpots = self._conditions_to_decor_rpots[conditions.fileformat]
            ave, vari = self._mbar.computeExpectations(values, rpots)
            aves.append(ave[0].astype(float))
            stds.append(np.sqrt(vari)[0].astype(float))

        return aves, stds
