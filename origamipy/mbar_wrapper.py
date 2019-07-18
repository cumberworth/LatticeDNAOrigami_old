import numpy as np
from pymbar import mbar

from origamipy import files
from origamipy import utility


class MBARWrapper:
    def __init__(self, decor_outs):
        self._decor_outs = decor_outs
        self._conditions_to_decor_rpots = {}

    def perform_mbar(self):
        print('Performing MBAR')
        rpots_matrix = self._calc_decorrelated_rpots_for_all_conditions()
        num_steps_per_condition = self._decor_outs.get_num_steps_per_condition()
        self._mbar = mbar.MBAR(rpots_matrix, num_steps_per_condition)

    def _calc_decorrelated_rpots_for_all_conditions(self):
        conditions_rpots = []
        decor_enes = self._decor_outs.get_concatenated_datatype('enes')
        decor_ops = self._decor_outs.get_concatenated_datatype('ops')
        for conditions in self._decor_outs.all_conditions:
            # What if I just want to do one rep?
            rpots = utility.calc_reduced_potentials(decor_enes, decor_ops,
                    conditions)
            conditions_rpots.append(rpots)
            # I should split this up if I want to separate mbar from calcs that use it
            self._conditions_to_decor_rpots[conditions.fileformat] = rpots

        return np.array(conditions_rpots)

    def calc_all_expectations(self, filebase):
        print('Calculating all expectation values')
        all_aves = []
        all_stds = []
        all_tags = self._decor_outs.all_conditions.condition_tags
        series_tags = self._decor_outs.all_series_tags
        for i, tag in enumerate(series_tags):
            print('Calculating expectation of {} ({} of {})'.format(tag, i,
                len(series_tags)))
            values = self._decor_outs.get_concatenated_series(tag)
            aves, stds = self._calc_expectations(values)
            all_tags.append(tag)
            all_aves.append(aves)
            all_stds.append(stds)

        all_conds = self._decor_outs.all_conditions.condition_to_characteristic_values
        all_conds = np.array(all_conds, dtype=float)
        aves_file = files.TagOutFile('{}.aves'.format(filebase))
        aves_file.write(all_tags, np.concatenate([all_conds,
                                                  np.array(all_aves).T], axis=1))
        stds_file = files.TagOutFile('{}.stds'.format(filebase))
        stds_file.write(all_tags, np.concatenate([all_conds,
                                                  np.array(all_stds).T], axis=1))

    def _calc_expectations(self, values):
        aves = []
        stds = []
        for conditions in self._decor_outs.all_conditions:

            # It would be more efficient to collect all timeseries of the same
            # conditions and run this once
            rpots = self._conditions_to_decor_rpots[conditions.fileformat]
            ave, vari = self._mbar.computeExpectations(values, rpots)
            aves.append(ave[0].astype(float))
            stds.append(np.sqrt(vari)[0].astype(float))

        return aves, stds

    def calc_expectation(self, values, conds):
        decor_enes = self._decor_outs.get_concatenated_datatype('enes')
        decor_ops = self._decor_outs.get_concatenated_datatype('ops')
        rpots = utility.calc_reduced_potentials(decor_enes, decor_ops,
                conds)
        ave, vari = self._mbar.computeExpectations(values, rpots)
        ave = ave[0].astype(float)
        std = np.sqrt(vari)[0].astype(float)

        return ave, std

    def calc_1d_lfes(self, reduced_conditions, filebase, xtag='temp'):
        print('Calculating all 1D LFEs')

        # This is bad
        all_conds = reduced_conditions.condition_to_characteristic_values
        all_tags = reduced_conditions.condition_tags
        xvar_i = all_tags.index(xtag)
        xvars = [c[xvar_i] for c in all_conds]
        series_tags = self._decor_outs.all_series_tags

        # Also bad
        for tag in ['tenergy', 'henthalpy', 'hentropy', 'stacking', 'bias']:
            series_tags.remove(tag)

        for i, tag in enumerate(series_tags):
            print('Calculating 1D LFEs of {} ({} of {})'.format(tag, i, len(series_tags)))
            values = self._decor_outs.get_concatenated_series(tag)
            bins = list(set(values))
            bins.sort()
            lfes, stds = self._calc_lfes(bins, values, reduced_conditions)

            # Ugly
            header = np.concatenate([['ops'], xvars])
            lfes_filebase = '{}_{}-lfes'.format(filebase, tag)

            lfes_file = files.TagOutFile('{}.aves'.format(lfes_filebase))
            lfes = self._hack_prepare_1d_lfe_series_for_write(lfes, bins)
            lfes_file.write(header, lfes)

            stds_file = files.TagOutFile('{}.stds'.format(lfes_filebase))
            stds = self._hack_prepare_1d_lfe_series_for_write(stds, bins)
            stds_file.write(header, stds)

    def _calc_lfes(self, bins, values, reduced_conditions):
        value_to_bin = {value: i for i, value in enumerate(bins)}
        bin_index_series = [value_to_bin[i] for i in values]
        bin_index_series = np.array(bin_index_series)
        all_lfes = []
        all_lfe_stds = []
        for conditions in reduced_conditions:
            rpots = self._conditions_to_decor_rpots[conditions.fileformat]
            lfes, lfe_stds = self._mbar.computePMF(
                rpots, bin_index_series, len(bins))
            all_lfes.append(lfes)
            all_lfe_stds.append(lfe_stds)

        return all_lfes, all_lfe_stds

    def calc_specified_2d_lfes(self, tag_pairs, reduced_conditions, filebase):
        print('Calculating 2D LFEs for all specified pairs')
        for i, tag_pair in enumerate(tag_pairs):
            tag1, tag2 = tag_pair
            print('Calculating 2D LFEs of {}-{} ({} of {})'.format(tag1, tag2, i, len(tag_pairs)))
            self.calc_2d_lfes(tag1, tag2, reduced_conditions, filebase)

    def calc_2d_lfes(self, tag1, tag2, reduced_conditions, filebase):
        print('Calculating 2D LFEs of {}-{}'.format(tag1, tag2))
        temps = [c.temp for c in reduced_conditions]
        decor_value_pairs = list(zip(self._decor_outs.get_concatenated_series(tag1),
                                 self._decor_outs.get_concatenated_series(tag2)))
        bins = list(set(decor_value_pairs))
        lfes, stds = self._calc_lfes(bins, decor_value_pairs, reduced_conditions)

        # Ugly
        header = np.concatenate([[tag1, tag2], temps])
        lfes_filebase = '{}_{}-{}-lfes'.format(filebase, tag1, tag2)

        lfes_file = files.TagOutFile('{}.aves'.format(lfes_filebase))
        lfes = self._hack_prepare_2d_lfe_series_for_write(lfes, bins)
        lfes_file.write(header, lfes)

        stds_file = files.TagOutFile('{}.stds'.format(lfes_filebase))
        stds = self._hack_prepare_2d_lfe_series_for_write(stds, bins)
        stds_file.write(header, stds)

    def _hack_prepare_1d_lfe_series_for_write(self, series, bins):
        bins = np.array(bins).reshape(len(bins), 1)
        return np.concatenate([bins, np.array(series).T], axis=1)

    def _hack_prepare_2d_lfe_series_for_write(self, series, bins):
        bins = np.array(bins).reshape(len(bins), 2)
        return np.concatenate([bins, np.array(series).T], axis=1)
