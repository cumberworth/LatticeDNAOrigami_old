#!/bin/env python

"""Calculate scaffold RMSD relative to a given configuration."""

import math
import argparse
import sys

import numpy as np
import scipy.constants

name = 's1'
run = 0
reps = 50

rmsdrmsds = []
for rep in reps:
    sdrmsd = []
    data = np.loadtxt('{}_run-{}_rep-{}_rmsd.dat'.format(name, run, rep), skiprows=1)
    mean = data.mean(axis=1)[1]
    for step in data:
        sdrmsd.append((mean - data[step][1])**2)

    rmsdrmsds.append(sdrmsd)

rmsdrmsds.mean(axis=0)

output_row_major('{}_run-{}_rmsdrmsd.dat'.format(name, run))
