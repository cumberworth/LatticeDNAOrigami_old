import numpy as np
from pymbar import mbar
from scipy.signal import argrelextrema
from scipy import interpolate
from scipy.optimize import minimize

from origamipy import files
from origamipy import utility


class MBARWrapper:
    def __init__(self, decor_outs):
        self._decor_outs = decor_outs

    def perform_mbar(self):
        rpots_matrix = self._calc_decorrelated_rpots_for_all_conditions()
        num_steps_per_condition = self._decor_outs.num_steps_per_condition
        self._mbar = mbar.MBAR(rpots_matrix, num_steps_per_condition)

    def calc_expectation(self, se_tag, conds):
        """Calculate expectation and std for given op and conditions."""
        series = self._decor_outs.get_concatenated_series(se_tag)
        decor_enes = self._decor_outs.get_concatenated_datatype('enes')
        decor_ops = self._decor_outs.get_concatenated_datatype('ops')
        decor_staples = self._decor_outs.get_concatenated_datatype('staples')
        rpots = utility.calc_reduced_potentials(
            decor_enes, decor_ops, decor_staples, conds)
        ave, std = self._mbar.computeExpectations(series, rpots)

        return ave[0].astype(float), std[0].astype(float)

    def calc_all_expectations(self, filebase, se_tags, all_conds):
        """Calculate expectation and std for given ops and conditions.

        Write both expectation values and standard deviations to file.
        """
        all_aves = []
        all_stds = []
        for se_tag in se_tags:
            aves = []
            stds = []
            for conds in all_conds:
                ave, std = self.calc_expectation(se_tag, conds)
                aves.append(ave)
                stds.append(std)

            all_aves.append(aves)
            all_stds.append(stds)

        if type(all_conds) == list:
            conds_tags = all_conds[0].condition_tags
            all_conds_values = []
            for conds in all_conds:
                values = conds.characteristic_values
                all_conds_values.extend(values)
        else:
            conds_tags = all_conds.condition_tags
            all_conds_values = all_conds.conditions_to_characteristic_values

        all_tags = np.concatenate([conds_tags, se_tags])
        all_conds_values = np.array(all_conds_values, dtype=float)

        conds_aves = np.concatenate(
            [all_conds_values, np.array(all_aves).T], axis=1)
        aves_file = files.TagOutFile('f{filebase}.aves')
        aves_file.write(all_tags, conds_aves)

        conds_stds = np.concatenate(
            [all_conds_values, np.array(all_stds).T], axis=1)
        stds_file = files.TagOutFile(f'{filebase}.stds')
        stds_file.write(all_tags, conds_stds)

    def calc_1d_lfes(self, se_tag, conds, filebase=None, xtag='temp'):
        """Calculate 1D LFEs and stds for given op and conditions.

        Write both LFEs and standard deviations to file.
        """
        series = self._decor_outs.get_concatenated_series(se_tag)
        bins = list(set(series))
        bins.sort()
        lfe, std = self._calc_lfes(bins, series, conds)

        if filebase != None:
            header = [se_tag, xtag]
            conds_tags = all_conds.condition_tags
            xvar = conds[xtag]

            lfes_filebase = f'{filebase}-{se_tag}'
            lfes_file = files.TagOutFile(f'{lfes_filebase}.aves')
            self._write_lfe_series_to_file(lfes_file, header, lfe, bins, 1)

            stds_file = files.TagOutFile(f'{lfes_filebase}.stds')
            self._write_lfe_series_to_file(stds_file, header, std, bins, 1)

        return lfe, std, bins

    def calc_all_1d_lfes(self, filebase, se_tags, all_conds, xtag='temp'):
        """Calculate 1D LFEs for given ops and conditions.

        Write both LFEs and standard deviations to file.
        """
        if type(all_conds) == list:
            conds_tags = all_conds[0].condition_tags
        else:
            conds_tags = all_conds.condition_tags
        xvars = [c[xtag] for c in all_conds]

        for se_tag in se_tags:
            lfes = []
            stds = []
            series = self._decor_outs.get_concatenated_series(se_tag)
            bins = list(set(series))
            bins.sort()
            for conds in all_conds:
                lfe, std = self._calc_lfes(bins, series, conds)
                lfes.append(lfe)
                stds.append(std)

            header = np.concatenate([[se_tag], xvars])
            lfes_filebase = f'{filebase}-{se_tag}'

            lfes_file = files.TagOutFile(f'{lfes_filebase}.aves')
            self._write_lfe_series_to_file(lfes_file, header, lfes, bins, 1)

            stds_file = files.TagOutFile(f'{lfes_filebase}.stds')
            self._write_lfe_series_to_file(stds_file, header, stds, bins, 1)

    def calc_2d_lfes(self, se_tag_pair, conds):
        """Calculate 2D LFEs and stds for given op and conditions.

        Write both LFEs and standard deviations to file.
        """
        set_tag_1 = se_tag_pair[0]
        set_tag_2 = se_tag_pair[1]
        series_1 = self._decor_outs.get_concatenated_series(se_tag_1)
        series_2 = self._decor_outs.get_concatenated_series(se_tag_2)
        series_pairs = list(zip(series_1, series_2))
        bins = list(set(series_pairs))
        lfe, std = self._calc_lfes(bins, series_pairs, conds)

        header = [se_tag, 'lfes']

        lfes_filebase = f'{filebase}-{se_tag}'
        lfes_file = files.TagOutFile(f'{lfes_filebase}.aves')
        self._write_lfe_series_to_file(lfes_file, header, lfe, bins, 2)

        stds_file = files.TagOutFile(f'{lfes_filebase}.stds')
        self._write_lfe_series_to_file(stds_file, header, std, bins, 2)

        return lfe, std, bins

    def calc_all_2d_lfes(self, filebase, se_tag_pairs, all_conds, xtag='temp'):
        """Calculate 2D LFEs for given ops and conditions.

        Write both LFEs and standard deviations to file.
        """
        conds_tags = reduced_conditions.condition_tags
        xvar_i = conds_tags.index(xtag)
        xvars = [c[xvar_i] for c in all_conds]

        for se_tag_pair in se_tag_pairs:
            set_tag_1 = se_tag_pair[0]
            set_tag_2 = se_tag_pair[1]
            lfes = []
            stds = []
            series_1 = self._decor_outs.get_concatenated_series(se_tag_1)
            series_2 = self._decor_outs.get_concatenated_series(se_tag_2)
            series_pairs = list(zip(series_1, series_2))
            bins = list(set(series_pairs))
            for conds in all_conds:
                lfes.append(lfe)
                stds.append(std)
                lfe, std = self._calc_lfes(bins, series_pairs, conds)

            header = np.concatenate([[se_tag1, se_tag2], xvars])
            lfes_filebase = f'{filebase}-{se_tag_1}-{se_tag_2}'

            lfes_file = files.TagOutFile(f'{lfes_filebase}.aves')
            self._write_lfe_series_to_file(lfes_file, header, lfes, bins, 1)

            stds_file = files.TagOutFile(f'{lfes_filebase}.stds')
            self._write_lfe_series_to_file(stds_file, header, stds, bins, 1)

    def estimate_melting_temp(self, conds, guess_temp):
        """Estimate the melting temperature assuming barrier exists.

        Find the global maximum that is not at the edge of the domain and then
        find minima on either side and minimize difference between them.
        """
        series = self._decor_outs.get_concatenated_series(
            'numfullyboundstaples')
        bins = list(set(series))
        bins.sort()
        lfes = self._calc_lfes(bins, series, conds)
        melting_temp = minimize(self._squared_barrier_diff, guess_temp, args=(
            bins, series, conds)).x[0]

        return melting_temp

    def _calc_decorrelated_rpots_for_all_conditions(self):
        conditions_rpots = []
        enes = self._decor_outs.get_concatenated_datatype('enes')
        ops = self._decor_outs.get_concatenated_datatype('ops')
        staples = self._decor_outs.get_concatenated_datatype('staples')
        for conds in self._decor_outs.all_conditions:

            # What if I just want to do one rep?
            rpots = utility.calc_reduced_potentials(enes, ops, staples, conds)
            conditions_rpots.append(rpots)

        return np.array(conditions_rpots)

    def _calc_lfes(self, bins, series, conds):
        enes = self._decor_outs.get_concatenated_datatype('enes')
        ops = self._decor_outs.get_concatenated_datatype('ops')
        staples = self._decor_outs.get_concatenated_datatype('staples')
        rpots = utility.calc_reduced_potentials(enes, ops, staples, conds)

        value_to_bin = {value: i for i, value in enumerate(bins)}
        bin_index_series = [value_to_bin[i] for i in series]
        bin_index_series = np.array(bin_index_series)
        lfes, stds = self._mbar.computePMF(
            rpots, bin_index_series, len(bins))

        return lfes, stds

    def _write_lfe_series_to_file(self, file, header, series, bins, dim):
        bins = np.array(bins).reshape(len(bins), dim)
        data = np.concatenate([bins, np.array(series).T], axis=1)
        file.write(header, data)

    def _squared_barrier_diff(self, temp, bins, series, conds):
        conds._conditions['temp'] = temp
        lfes, stds = self._calc_lfes(bins, series, conds)
        barrier_i = find_barrier(lfes)
        minima = find_minima(lfes, barrier_i)

        return (minima[0] - minima[1])**2


# Maybe these should be somewhere else?
def find_barrier(lfes):
    maxima_i = argrelextrema(lfes, np.greater)[0]
    if len(maxima_i) == 0:
        print('No barrier detected')
        raise Exception

    maxima = lfes[maxima_i]
    maximum_i = maxima_i[maxima.argmax()]

    return maximum_i


def find_minima(lfes, maximum_i):
    lower_lfes = lfes[:maximum_i]
    upper_lfes = lfes[maximum_i:]

    return (lower_lfes.min(), upper_lfes.min())


def calc_forward_barrier_height(lfes):
    barrier_i = find_barrier(lfes)
    minima = find_minima(lfes, barrier_i)

    return lfes[barrier_i] - minima[0]
