#!/usr/bin/env python

from distutils.core import setup
from Cython.Build import cythonize

setup(name='origami_sytem,
        'ext_miodules=cythonize('origami_system.pyx'))
