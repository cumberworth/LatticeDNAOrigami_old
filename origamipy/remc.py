"""Analysis of REMC simulations"""

import collections

from origamipy import files


def deconvolute_remc_outputs(all_exchange_params, fileinfo, filetypes):
    swapfile = files.SwapInpFile(fileinfo.inputdir, fileinfo.filebase)
    for filetype in filetypes:
        f_collection = FileCollection(all_exchange_params, fileinfo, filetype)
        for threads_to_replicas in swapfile:
            f_collection.deconvolute_and_write_step(threads_to_replicas)


class FileCollection:
    """Read from thread files and write to replica files"""

    def __init__(self, all_exchange_params, fileinfo, ext):
        self._fileinfo = fileinfo
        self._ext = ext
        self._all_exchange_params = all_exchange_params
        self._num_threads = len(all_exchange_params)
        self._thread_files = []
        self._replica_files = []
        self._filetype = files.UnparsedStepInpFile
        self._num_header_lines = 0

        self._set_filetype()
        self._open_thread_files()
        self._open_replica_files()
        self._write_headers()

    def deconvolute_and_write_step(self, threads_to_replicas):
        for replica, thread in enumerate(threads_to_replicas):
            step = next(self._thread_files[thread])
            self._replica_files[replica].write(step)

    def _set_filetype(self):
        if self._ext in ['trj', 'vcf']:
            self._num_header_lines = 0
            self._filetype = files.UnparsedMultiLineStepInpFile
        elif self._ext in ['ores', 'states', 'staples',
                           'staplestates']:
            self._num_header_lines = 0
            self._filetype = files.UnparsedSingleLineStepInpFile
        elif self._ext in ['times', 'ene', 'ops']:
            self._num_header_lines = 1
            self._filetype = files.UnparsedSingleLineStepInpFile
        else:
            raise NotImplementedError

    def _open_thread_files(self):
        for thread in range(self._num_threads):
            filename = '{}/{}-{}.{}'.format(self._fileinfo.inputdir,
                                            self._fileinfo.filebase, thread,
                                            self._ext)
            thread_file = self._filetype(filename, self._num_header_lines)
            self._thread_files.append(thread_file)

    def _open_replica_files(self):
        for params in self._all_exchange_params:
            filename = '{}/{}-{}.{}'.format(self._fileinfo.outputdir,
                                            self._fileinfo.filebase, params,
                                            self._ext)
            self._replica_files.append(open(filename, 'w'))

    def _write_headers(self):
        for thread_f, replica_f in zip(self._thread_files,
                                       self._replica_files):
            replica_f.write(thread_f.header)
