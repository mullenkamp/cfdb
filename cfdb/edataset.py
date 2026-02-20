#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDataset classes and open function for S3-backed datasets via ebooklet.
This module is only imported when ebooklet is installed.
"""
import ebooklet
from typing import Union
import pathlib

from . import utils
from .main import Dataset


class EDataset(Dataset):
    """

    """
    def changes(self):
        """
        Return a Change object of the changes that have occurred during this session.
        """
        return self._blt.changes()

    def delete_remote(self):
        """
        Completely delete the remote dataset, but keep the local dataset.
        """
        self._blt.delete_remote()

    def copy_remote(self, remote_conn: ebooklet.S3Connection):
        """
        Copy the entire remote dataset to another remote location. The new location must be empty.
        """
        self._blt.copy_remote(remote_conn)


class EGrid(EDataset):
    """

    """

class ETimeSeriesOrtho(Dataset):
    """

    """


def open_edataset(remote_conn: Union[ebooklet.S3Connection, str, dict],
                  file_path: Union[str, pathlib.Path],
                  flag: str = "r",
                  dataset_type: str='grid',
                  compression: str='zstd',
                  compression_level: int=1,
                  num_groups: int = None,
                  **kwargs):
    """
    Open a cfdb that is linked with a remote S3 database.

    Parameters
    -----------
    remote_conn : S3Connection, str, or dict
        The object to connect to a remote. It can be an S3Connection object, an http url string, or a dict with the parameters for initializing an S3Connection object.

    file_path : str or pathlib.Path
        It must be a path to a local file location. If you want to use a tempfile, then use the name from the NamedTemporaryFile initialized class.

    flag : str
        Flag associated with how the file is opened according to the dbm style. See below for details.
    dataset_type: str
        The dataset type to be opened. The current options are "grid" and "ts_ortho". The default is "grid". See below for a description.
    compression: str
        The compression algorithm used for compressing all data. Must be either zstd or lz4. The option zstd has a really good combo of compression ratio to speed, while lz4 has a stronger emphasis on speed (and is lightning fast). Default is zstd.
    compression_level: int or None
        The compression level used by the compression algorithm. Setting this to None will d=used the deafults, which is 1 for both compression options.
    num_groups : int or None
        The number of groups for grouped S3 object storage. Required when creating a new database (flag='n'). For existing databases, this value is read from S3 metadata and the user-provided value is ignored.
    kwargs
        Any kwargs that can be passed to ebooklet.open.

    Returns
    -------
    cfdb.EDataset

    The optional *flag* argument can be:
    +---------+-------------------------------------------+
    | Value   | Meaning                                   |
    +=========+===========================================+
    | ``'r'`` | Open existing database for reading only   |
    |         | (default)                                 |
    +---------+-------------------------------------------+
    | ``'w'`` | Open existing database for reading and    |
    |         | writing                                   |
    +---------+-------------------------------------------+
    | ``'c'`` | Open database for reading and writing,    |
    |         | creating it if it doesn't exist           |
    +---------+-------------------------------------------+
    | ``'n'`` | Always create a new, empty database, open |
    |         | for reading and writing                   |
    +---------+-------------------------------------------+

    The dataset_type argument determines the dimension/coordinate data structure:
        grid: The standard CF conventions dimensions/coordinates. Each coordinate must be unique and increasing in ascending order. Each coordinate represents a single axis (i.e. x, y, z, t). The z axis is currently optional.
        ts_ortho: The a special time series coordinate structure representing the `Orthogonal multidimensional array representation of time series <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#_orthogonal_multidimensional_array_representation_of_time_series>`_. The Geometry dtype must represent the xy axis. The time coordinate is the same as the "grid" time coordinate. The z axis is currently optional.
    """
    if 'n_buckets' not in kwargs:
        kwargs['n_buckets'] = utils.default_n_buckets

    fp = pathlib.Path(file_path)
    fp_exists = fp.exists()
    open_blt = ebooklet.open(remote_conn, file_path, flag, num_groups=num_groups, **kwargs)

    if (not fp_exists or flag == 'n') and open_blt.writable:
        create = True
    else:
        create = False

    if dataset_type.lower() == 'grid':
        return EGrid(fp, open_blt, create, compression, compression_level, 'grid')
    else:
        raise TypeError('The only option for the dataset type is "grid".')
