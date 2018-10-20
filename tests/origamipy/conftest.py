"""Shared fixtures for testing origamipy"""

import pytest

from origamipy import io

@pytest.fixture
def four_domain_struct_inp_file():
    return io.JSONStructInpFile('tests/data/four_unbound.json')

print('HERE')
