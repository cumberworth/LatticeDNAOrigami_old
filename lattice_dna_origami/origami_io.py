#!/usr/env python

"""IO classes for origami simulations."""

import json

import h5py
import numpy as np

from lattice_dna_origami.origami_system import *

class OutputFile:
    """Base output file class to allow check_and_write to be shared."""

    def check_and_write(self, origami_system, step):
        """Check property write frequencies and write accordingly."""
        if value_is_multiple(step, self._config_write_freq):
            self.write_configuration(origami_system, step)
        if value_is_multiple(step, self._count_write_freq):
            self.write_staple_domain_count(origami_system)
        if value_is_multiple(step, self._energy_write_freq):
            self.write_energy(origami_system)

        # Move types and acceptances


class JSONOutputFile(OutputFile):
    """JSON output file class."""

    def __init__(self, filename, origami_system, config_write_freq=1):
        self._filename = filename
        self._config_write_freq = config_write_freq

        self.json_origami = {'origami':{'identities':{}, 'configurations':[]}}
        self.json_origami['origami']['identities'] = origami_system.identities
        self.json_origami['origami']['sequences'] = origami_system.sequences
        self.json_origami['origami']['cyclic'] = origami_system.cyclic

    def write_seed(self, seed):
        self.json_origami['origami']['seed'] = seed

    def write_configuration(self, origami_system, step):
        self.json_origami['origami']['configurations'].append({})
        current_config = self.json_origami['origami']['configurations'][-1]
        current_config['step'] = step
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

    def close(self):
        """Perform cleanup."""
        pass


class JSONInputFile:
    """Input file taking json formatted origami system files in constructor."""

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


class HDF5OutputFile(OutputFile):
    """HDF5 output file class.

    Custom format; not compatable with VMD (not H5MD).
    """

    def __init__(self, filename, origami_system, config_write_freq=1,
                count_write_freq=0, energy_write_freq=1):
        self.hdf5_origami = h5py.File(filename, 'w')
        self.hdf5_origami.create_group('origami')
        self.filename = filename
        self._config_write_freq = config_write_freq
        self._config_writes = 0
        self._count_write_freq = count_write_freq
        self._count_writes = 0
        self._energy_write_freq = energy_write_freq
        self._energy_writes = 0

        # HDF5 does not allow variable length lists; fill with 0
        max_domains = 0
        for domain_identities in origami_system.identities:
            domains = len(domain_identities)
            if domains > max_domains:
                max_domains = domains

        filled_identities = []
        for domain_identities in origami_system.identities:
            remainder = max_domains - len(domain_identities)
            filled_identities.append(domain_identities[:])
            for i in range(remainder):
                filled_identities[-1].append(0)

        self.hdf5_origami.create_dataset('origami/identities',
                data=filled_identities)

        # Fill attributes
        self.hdf5_origami.attrs['cyclic'] = origami_system.cyclic
        self.hdf5_origami.attrs['temp'] = origami_system.temp
        self.hdf5_origami.attrs['config_write_freq'] = config_write_freq
        self.hdf5_origami.attrs['count_write_freq'] = count_write_freq

        # HDF5 does not allow lists of strings
        linear_seqs = [j for i in origami_system.sequences for j in i]
        sequences = np.array(linear_seqs, dtype='a')
        self.hdf5_origami.create_dataset('origami/sequences',
                data=sequences)

        # Setup configuration datasets
        if config_write_freq > 0:
            self.hdf5_origami.create_group('origami/configurations')
            for chain in origami_system.chains:
                self._create_chain(chain)

        # Setup up analysis datasets:
        if self._count_write_freq > 0:
            self.hdf5_origami.create_dataset('origami/staple_domain_count',
                    (1, 2),
                    maxshape=(None, 2),
                    chunks=(1, 2),
                    dtype=int)

        if self._energy_write_freq > 0:
            self.hdf5_origami.create_dataset('origami/energies',
                    (1, 1),
                    maxshape=(None, 1),
                    chunks=(1, 1),
                    dtype=float)

        # Create array of chains present at each step
        dt = h5py.special_dtype(vlen=np.dtype('int32'))
        self.hdf5_origami.create_dataset('origami/chain_ids',
                (1,),
                maxshape=(None,),
                chunks=(1,),
                dtype=dt)

    def write_seed(self, seed):

        # h5py docs recommend wrapping byte strings in np.void
        self.hdf5_origami['origami'].attrs['seed'] = np.void(seed)

    def write_staple_domain_count(self, origami_system):
        write_index = self._count_writes
        self._count_writes += 1
        num_staples = origami_system.num_staples
        num_domains = origami_system.num_bound_domains
        count_key = 'origami/staple_domain_count'
        self.hdf5_origami[count_key].resize(self._count_writes, axis=0)
        self.hdf5_origami[count_key][write_index] = (num_staples, num_domains)

    def write_energy(self, origami_system):
        write_index = self._energy_writes
        self._energy_writes += 1
        energy = origami_system.energy
        energy_key = 'origami/energies'
        self.hdf5_origami[energy_key].resize(self._energy_writes, axis=0)
        self.hdf5_origami[energy_key][write_index] = energy

    def write_configuration(self, origami_system, step):
        write_index = self._config_writes
        self._config_writes += 1
        chain_ids = []
        for chain in origami_system.chains:
            chain_index = chain['index']
            chain_ids.append(chain_index)
            base_key = 'origami/configurations/{}'.format(chain_index)
            try:
                self.hdf5_origami[base_key]
            except KeyError:
                self._create_chain(chain)

            step_key = base_key + '/step'
            pos_key = base_key + '/positions'
            orient_key = base_key + '/orientations'
            self.hdf5_origami[step_key].resize(self._config_writes, axis=0)
            self.hdf5_origami[step_key][write_index] = step
            self.hdf5_origami[pos_key].resize(self._config_writes, axis=0)
            self.hdf5_origami[pos_key][write_index] = chain['positions']
            self.hdf5_origami[orient_key].resize(self._config_writes, axis=0)
            self.hdf5_origami[orient_key][write_index] = chain['orientations']

        self.hdf5_origami['origami/chain_ids'].resize(self._config_writes, axis=0)
        self.hdf5_origami['origami/chain_ids'][write_index] = chain_ids

    def close(self):
        """Perform any cleanup."""
        self.hdf5_origami.close()

    def _create_chain(self, chain):
        chain_length = len(chain['positions'])
        chain_index = chain['index']
        base_key = 'origami/configurations/{}'.format(chain_index)
        step_key = base_key + '/step'
        pos_key = base_key + '/positions'
        orient_key = base_key + '/orientations'

        self.hdf5_origami.create_group(base_key)
        self.hdf5_origami[base_key].attrs['index'] = (chain_index)
        self.hdf5_origami[base_key].attrs['identity'] = (chain['identity'])
        self.hdf5_origami.create_dataset(step_key,
                (1, 1),
                chunks=(1, 1),
                maxshape=(None, 1),
                dtype='i')
        self.hdf5_origami.create_dataset(pos_key,
                (1, chain_length, 3),
                chunks=(1, chain_length, 3),
                maxshape=(None, chain_length, 3),
                compression='gzip',
                dtype='i')
        self.hdf5_origami.create_dataset(orient_key,
                (1, chain_length, 3),
                chunks=(1, chain_length, 3),
                maxshape=(None, chain_length, 3),
                compression='gzip',
                dtype='i')


class HDF5InputFile:
    """Input file taking hdf5 formatted origami system files in constructor."""

    def __init__(self, filename):
        hdf5_origami = h5py.File(filename, 'r')

        self._filename = filename
        self._hdf5_origami = hdf5_origami
        self.config_write_freq = hdf5_origami.attrs['config_write_freq']
        self.count_write_freq = hdf5_origami.attrs['config_write_freq']

    @property
    def cyclic(self):
        return self._hdf5_origami.attrs['cyclic']

    @property
    def identities(self):
        """Standard format for passing origami domain identities."""

        # HDF5 does not allow variable length lists; fill with 0
        raw = np.array(self._hdf5_origami['origami/identities']).tolist()
        identities = []
        for raw_domain_identities in raw:
            identities.append([i for i in raw_domain_identities if i != 0])

        return identities

    @property
    def sequences(self):
        """Standard format for passing origami system sequences.

        Only includes sequences of scaffold in 5' to 3' orientation.
        """

        # H5py outputs as type 'S', need type 'U'
        identities = self.identities
        seqs = []
        linear_i = 0
        linear_seqs = np.array(self._hdf5_origami['origami/sequences']).astype(
                'U').tolist()
        for c_i, c_idents in enumerate(identities):
            for d_i in range(len(c_idents)):
                seqs[c_i][d_i] = linear_seqs[linear_i]
                linear_i += 1

        return seqs

    @property
    def temp(self):
        return self._hdf5_origami.attrs['temp']

    @property
    def steps(self):
        return len(self._hdf5_origami['origami/chain_ids'])

    @property
    def staple_domain_counts(self):
        return self._hdf5_origami['origami/staple_domain_count']

    @property
    def energy(self):
        return self._hdf5_origami['origami/energies'][:].flatten()

    def chains(self, step):
        """Standard format for passing chain configuration."""

        # For consistency, do not autoconvert step to index
        #step = step / self.config_write_freq
        chains = []
        chain_ids = self._hdf5_origami['origami/chain_ids'][step]
        base_key = 'origami/configurations'
        for chain in chain_ids:
            chain_key = base_key + '/' + str(chain)
            chain_group = self._hdf5_origami[chain_key]
            chain = {}
            chain['index'] = chain_group.attrs['index']
            chain['identity'] = chain_group.attrs['identity']
            chain['positions'] = chain_group['positions'][step].tolist()
            chain['orientations'] = chain_group['orientations'][step].tolist()
            chains.append(chain)

        return chains

    def close(self):
        self._hdf5_origami.close()



class PlainTextTrajFile:
    """Custom plain text trajectory file for cpp version.
    
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
        chain['positions'] = pos_row_major.reshape(len(pos_row_major) // 3, 3)
        line_i += 1
        ore_row_major = np.array(lines[line_i].split()).astype(int)
        chain['orientations'] = ore_row_major.reshape(len(ore_row_major) // 3, 3)
        line_i += 1
        self._steps[-1].append(chain)
        return  line_i
