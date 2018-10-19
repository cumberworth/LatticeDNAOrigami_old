"""Tests for origamipy.io"""

import pytest

from origamipy import io

@pytest.fixture
def four_domain_struct_inp_file():
    return io.JSONStructInpFile('tests/data/four_unbound.json')

@pytest.fixture
def txt_traj_inp_file(four_domain_struct_inp_file):
    return io.TxtTrajInpFile('tests/data/four.trj', four_domain_struct_inp_file)

def test_get_next_config_from_TxtTrajInpFile(txt_traj_inp_file):
    next(txt_traj_inp_file)

def test_for_loop_over_TxtTrajInpFile(txt_traj_inp_file):
    num_chains = [4, 5, 4]
    for i, chains in enumerate(txt_traj_inp_file):
        assert len(chains) == num_chains[i]

    assert i == 2

    for i, chains in enumerate(txt_traj_inp_file):
        assert len(chains) == num_chains[i]

    assert i == 2
