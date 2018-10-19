"""Tests for origamipy.remc"""

import pytest

from origamipy import remc
from origamipy import io

@pytest.fixture
def temps():
    return []

@pytest.fixture
def stack_mults():
    return []

@pytest.fixture
def fileinfo(tmpdir):
    return io.FileInfo('tests/data', tmpdir, 'four-2dremc')

@pytest.fixture
def all_exchange_params(temps, stack_mults):
    remc.create_exchange_params(temps, stack_mults)

def test_deconvolute_remc_trj(all_exchange_params, fileinfo):
    filetypes = ['trj']
    remc.deconvolute_remc_outputs(all_exchange_params, fileinfo, filetypes)
    for params in all_exchange_params:
        params = remc.exchange_params_to_subfile_string(params)
        output_filename = '{}/{}-{}.trj'.format(fileinfo.outputdir,
                                                fileinfo.filebase, params)
        expected_filename = '{}/{}-{}.trj'.format(fileinfo.inputdir,
                                                  fileinfo.filebase, params)
        assert open(output_filename).read() == open(expected_filename).read()
