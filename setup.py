"""Setup script for analysis module and related scripts LatticeDNAOrigami"""

import glob
from setuptools import setup

setup(
    name='origamipy',
    packages=['origamipy'],
    scripts=glob.glob('scripts/*/*.py')
)
