#!/usr/env python

"""IO classes for origami system information, topologies, and configurations"""

import numpy as np
import json
 

class JSONInputFile:
    """JSON input file for system information, topology, and configuration.
    
    Can contain multiple configurations.
    """

    def __init__(self, filename):
        json_origami = json.load(open(filename))

        self._filename = filename
        self._json_origami = json_origami

    @property
    def cyclic(self):
        return self._json_origami['origami']['cyclic']

    @property
    def identities(self):
        """Standard format for passing origami domain identities."""
        return self._json_origami['origami']['identities']

    @property
    def sequences(self):
        """Standard format for passing origami domain sequences.

        Only includes sequences of scaffold in 5' to 3' orientation.
        """
        return self._json_origami['origami']['sequences']

    def chains(self, step):
        """Standard format for passing chain configuration."""
        return self._json_origami['origami']['configurations'][step]['chains']

    def close(self):
        pass


class JSONOutputFile:
    """JSON output file for system information, topology, and configuration."""

    def __init__(self, filename, origami_system):
        self._filename = filename

        self.json_origami = {'origami':{'identities':{}, 'configurations':[]}}
        self.json_origami['origami']['identities'] = origami_system.identities
        self.json_origami['origami']['sequences'] = origami_system.sequences
        self.json_origami['origami']['cyclic'] = origami_system.cyclic

    def write(self, origami_system):
        self.json_origami['origami']['configurations'].append({})
        current_config = self.json_origami['origami']['configurations'][-1]

        # Step should probably be changed as this is no longer being used by a
        # a simulation class
        current_config['step'] = 0
        current_config['chains'] = []
        for chain in origami_system.chains:
            current_config['chains'].append({})
            json_chain = current_config['chains'][-1]
            json_chain['identity'] = chain['identity']
            json_chain['index'] = chain['index']
            json_chain['positions'] = chain['positions']
            json_chain['orientations'] = chain['orientations']

        json.dump(self.json_origami, open(self._filename, 'w'), indent=4,
                    separators=(',', ': '))


class PlainTextTrajFile:
    """Plain text trajectory input file.
    
    Requires system information be provided through an input file."""

    def __init__(self, traj_filename, input_file):
        self._input_file = input_file
        self._traj_filename = traj_filename

        # Interval that configurations have been written to file
        self._write_interval = 1

        # List of chains at each steps
        self._steps = []
        self._parse()

    @property
    def cyclic(self):
        return self._input_file.cyclic

    @property
    def sequences(self):
        return self._input_file.sequences

    @property
    def identities(self):
        return self._input_file.identities

    @property
    def steps(self):
        return len(self._steps)

    def chains(self, step):
        """Standard format for passing chain configuration."""
        return self._steps[step]

    def _parse(self):
        """ Parse trajectory file."""
        with open(self._traj_filename) as inp:
            lines = inp.read().split('\n')

        # File will start with step number
        self._write_interval = int(lines[0])
        line_i = 1
        eof = False
        while not eof:
            eoc = False
            self._steps.append([])
            while not (eoc or eof):
                line_i = self._parse_chain(lines, line_i)
                if line_i == len(lines) - 2:
                    eof = True
                elif lines[line_i] == '':
                    eoc = True
            line_i += 2

    def _parse_chain(self, lines, line_i):
        chain = {}
        chain_ids = [int(i) for i in lines[line_i].split()]
        chain['index'] = chain_ids[0]
        chain['identity'] = chain_ids[1]
        line_i += 1
        pos_row_major = np.array(lines[line_i].split()).astype(int)
        chain['positions'] = pos_row_major.reshape(len(pos_row_major) // 3, 3).tolist()
        line_i += 1
        ore_row_major = np.array(lines[line_i].split()).astype(int)
        chain['orientations'] = ore_row_major.reshape(len(ore_row_major) // 3, 3).tolist()
        line_i += 1
        self._steps[-1].append(chain)
        return  line_i


class PlainTextTrajOutFile:
    """Plain text trajectory output file."""

    def __init__(self, filename):
        self.file = open(filename, 'w')

    def write_config(self, chains, step):
        self.file.write('{}\n'.format(step))
        for chain in chains:
            self.file.write('{} '.format(chain['index']))
            self.file.write('{}\n'.format(chain['identity']))
            for pos in chain['positions']:
                for comp in pos:
                    self.file.write('{} '.format(comp))
            self.file.write('\n')
            for ore in chain['orientations']:
                for comp in ore:
                    self.file.write('{} '.format(comp))
            self.file.write('\n')
        self.file.write('\n')
