#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDataset classes and open function for S3-backed datasets via ebooklet.
This module is only imported when ebooklet is installed.
"""
from __future__ import annotations

import ebooklet
import msgspec
from typing import Union
import pathlib

from . import utils
from .main import Dataset

## "Argument not given" for open_edataset's storage-mode arguments: only given
## arguments are forwarded, so ebooklet's own defaults (inherit the remote's
## mode and recorded group_bytes; grouped for a new dataset) apply otherwise.
_NOT_GIVEN = object()


class EDataset(Dataset):
    """

    """
    def __init__(self, file_path, open_blt, create, compression, compression_level, dataset_type):
        super().__init__(file_path, open_blt, create, compression, compression_level, dataset_type)
        # Only write metadata when the remote flag actually flips (dataset creation, or linking a locally-created dataset to a remote). An unconditional write would bump the timestamp and push unchanged metadata every session.
        if self.writable and not self._sys_meta.remote:
            self._sys_meta.remote = True
            self._blt.set_metadata(msgspec.to_builtins(self._sys_meta))

    def changes(self):
        """
        Return a Change object of the changes that have occurred during this session.

        Flushes in-memory state (variable definitions and attributes) to the local file first, so the changelog reflects exactly what a push would publish.
        """
        self.sync()
        return self._blt.changes()

    def delete_remote(self):
        """
        Completely delete the remote dataset, but keep the local dataset.
        """
        self._blt.delete_remote()

    def push(self, force_push=False):
        """
        Push local changes to the remote. Safe to call mid-session: the current variable definitions and attributes are flushed to the local file before pushing.

        Returns an ebooklet PushResult: `updated` (the remote changed), `failures`
        (failed keys/groups -> error strings; the pending changes are retained for
        retry), and `bool(result)` is True only for a fully-successful push that
        changed the remote. Use force_push=True to retry after a partial failure.
        """
        return self.changes().push(force_push=force_push)

    def copy_remote(self, remote_conn: ebooklet.S3Connection):
        """
        Copy the entire remote dataset to another remote location. The new location must be empty.
        """
        self._blt.copy_remote(remote_conn)


class EGrid(EDataset):
    """

    """

class ETimeSeriesOrtho(EDataset):
    """

    """

class ETimeSeriesForecast(EDataset):
    """
    S3-backed (point, forecast_reference_time, forecast_period) station forecasts.
    """

class EGridForecast(EDataset):
    """
    S3-backed (x, y, forecast_reference_time, forecast_period) gridded forecasts.
    """


def open_edataset(remote_conn: Union[ebooklet.S3Connection, str, dict],
                  file_path: Union[str, pathlib.Path],
                  flag: str = "r",
                  dataset_type: str='grid',
                  compression: str=utils.default_compression,
                  compression_level: int=None,
                  *,
                  group_bytes=_NOT_GIVEN,
                  lock_timeout: int = 300,
                  force_lock: bool = False,
                  num_groups=_NOT_GIVEN,
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
        Flag associated with how the file is opened according to the dbm style.

        - ``'r'`` -- Open existing database for reading only (default).
        - ``'w'`` -- Open existing database for reading and writing.
        - ``'c'`` -- Open database for reading and writing, creating it if it doesn't exist.
        - ``'n'`` -- Always create a new, empty database, open for reading and writing.

        A dataset "exists" if it exists locally OR remotely: with ``'w'``/``'c'``, a fresh local file path attaches to an existing remote dataset (its structure is pulled on demand) rather than creating a new one. A new dataset is only created when neither exists (or with ``'n'``, which always creates new).
    dataset_type : str
        The dataset type when CREATING a new dataset. Default is ``'grid'``. Existing datasets (local or remote) always open with their stored type (and the matching class); this parameter is then ignored.

        - ``'grid'`` -- The standard CF conventions dimensions/coordinates. Each coordinate must be unique and increasing in ascending order. Each coordinate represents a single axis (i.e. x, y, z, t). The z axis is currently optional.
        - ``'ts_ortho'`` -- A special time series coordinate structure representing the orthogonal multidimensional array representation of time series. Designed for time series data with sparse geometries (e.g. station time series data). The Geometry dtype must represent the xy axis. The z axis is currently optional.
    compression : str
        The compression for all chunks, used only when a NEW dataset is created: attaching to an existing local or remote dataset always uses the compression it recorded, whatever is passed here. One of ``'zstd_shuffle'`` (default), ``'zstd'``, ``'lz4_shuffle'`` or ``'lz4'``; see ``open_dataset``.
    compression_level : int or None
        The compression level. None uses the defaults, which is 1 for every compression option.
    group_bytes : int, None, or omitted
        The remote storage mode (ebooklet >= 0.11). An int: grouped storage - at each push the chunks new to the remote are packed, in the order they were written, into group objects of up to group_bytes bytes, so appending to a dataset uploads only the new chunks (plus at most one partly filled group). None: per-key storage, one remote object per chunk.
        Omitted: an existing remote keeps its mode and the group_bytes it records (whatever its last push packed with); a NEW dataset is grouped at ebooklet.DEFAULT_GROUP_BYTES (32 MiB). Passing another int packs new chunks to it from then on and is recorded in place of the old value. Writing chunks in the order they will usually be read together (e.g. time band by time band) keeps them in the same groups. See ebooklet's "Grouped Storage".
    lock_timeout : int
        Maximum time in seconds to wait for the write lock when opening for write. Default is 300 (5 minutes). Only applies when flag is not ``'r'``. Raises ``TimeoutError`` if the lock cannot be acquired within the timeout.
    force_lock : bool
        If True, break any existing write locks before acquiring. Use this to recover from stale locks left by crashed processes. Default is False.
    num_groups : None
        Removed (hash grouping, ebooklet < 0.11). Forwarded to ebooklet for one release: an explicit num_groups=None keeps its old meaning, per-key storage; an int raises ValueError - use group_bytes.
    **kwargs
        Any kwargs that can be passed to ``ebooklet.open_ebooklet``.

    Returns
    -------
    cfdb.EDataset
    """
    if 'n_buckets' not in kwargs:
        kwargs['n_buckets'] = utils.default_n_buckets
    ## Forward the storage-mode arguments only when given: omitted, ebooklet
    ## inherits an existing remote's mode and recorded group_bytes (and
    ## defaults a new one to grouped).
    if group_bytes is not _NOT_GIVEN:
        kwargs['group_bytes'] = group_bytes
    if num_groups is not _NOT_GIVEN:
        kwargs['num_groups'] = num_groups

    fp = pathlib.Path(file_path)
    open_blt = ebooklet.open_ebooklet(remote_conn, file_path, flag, lock_timeout=lock_timeout, force_lock=force_lock, **kwargs)

    try:
        # Create only when no dataset exists anywhere: get_metadata() transparently checks the remote as well as the local file, so a fresh local file attaches to an existing remote dataset instead of silently creating a new (empty) one over it. flag 'n' always creates new.
        meta = None if flag == 'n' else open_blt.get_metadata()
        if flag == 'n':
            create = open_blt.writable
        else:
            create = open_blt.writable and meta is None

        ## The class follows the STORED dataset_type for existing datasets; the
        ## dataset_type parameter only applies at creation. (meta can be None on
        ## a read-only open of a corrupt/empty dataset - fall through to the
        ## param so Dataset.__init__ raises its clear msgspec ValidationError.)
        if create or meta is None:
            dt = dataset_type.lower()
        else:
            dt = meta['dataset_type']

        if dt == 'grid':
            return EGrid(fp, open_blt, create, compression, compression_level, 'grid')
        elif dt == 'ts_ortho':
            return ETimeSeriesOrtho(fp, open_blt, create, compression, compression_level, 'ts_ortho')
        elif dt == 'ts_forecast':
            return ETimeSeriesForecast(fp, open_blt, create, compression, compression_level, 'ts_forecast')
        elif dt == 'grid_forecast':
            return EGridForecast(fp, open_blt, create, compression, compression_level, 'grid_forecast')
        else:
            raise TypeError(
                f'dataset_type must be one of "grid", "ts_ortho", "ts_forecast" or '
                f'"grid_forecast"; got {dt!r}. If this dataset was written by a newer cfdb, '
                f'upgrade cfdb and cfdb-models.'
            )
    except BaseException:
        open_blt.close()
        raise
