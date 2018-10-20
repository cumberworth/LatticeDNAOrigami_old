"""Tests for origamipy.remc"""

import pytest

from origamipy import remc
from origamipy import io

@pytest.fixture
def temps():
    return [336, 342, 348, 354]

@pytest.fixture
def stack_mults():
    return [1.0, 0.6, 0.3, 0.0]

@pytest.fixture
def fileinfo(tmpdir):
    return io.FileInfo('tests/data', tmpdir, 'four-2dremc')

@pytest.fixture
def all_exchange_params(temps, stack_mults, four_domain_struct_inp_file):
    return remc.create_exchange_params(temps, stack_mults)

def test_deconvolute_remc_trj(all_exchange_params, fileinfo,
                              four_domain_struct_inp_file):
    filetypes = ['trj']
    remc.deconvolute_remc_outputs(all_exchange_params, fileinfo, filetypes)
    params = '342-0.0'
    output_filename = '{}/{}-{}.trj'.format(fileinfo.outputdir,
                                              fileinfo.filebase, params)
    output_trj_file = io.TxtTrajInpFile(output_filename,
                                   four_domain_struct_inp_file)
    original_filename = '{}/{}-{}.trj'.format(fileinfo.inputdir,
                                              fileinfo.filebase, 7)
    original_trj_file = io.TxtTrajInpFile(original_filename,
                                       four_domain_struct_inp_file)
    output_chains = output_trj_file.get_chains(2)
    original_chains = original_trj_file.get_chains(2)
    assert output_chains == original_chains
