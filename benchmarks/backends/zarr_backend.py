import shutil
from pathlib import Path

import numpy as np
import zarr

from benchmarks.backends.base import BackendBase


class ZarrBackend(BackendBase):
    name = 'zarr'

    def setup(self, base_dir: Path, data_config: dict):
        self.base_dir = base_dir
        self.data = data_config
        self._counter = 0

    def _next_path(self):
        self._counter += 1
        return self.base_dir / f'bench_{self._counter}.zarr'

    def write_full(self) -> Path:
        path = self._next_path()
        data = self.data
        chunk_shape = data['chunk_shape']

        store = zarr.storage.LocalStore(path)
        root = zarr.group(store=store, overwrite=True)

        root.create_array('latitude', data=data['lat_data'], chunks=(chunk_shape[0],))
        root.create_array('longitude', data=data['lon_data'], chunks=(chunk_shape[1],))

        time_int = (data['time_data'] - np.datetime64('1970-01-01', 'D')).astype('int64')
        root.create_array('time', data=time_int, chunks=(chunk_shape[2],))

        root.create_array(
            'temperature',
            data=data['var_data'],
            chunks=chunk_shape,
            compressors=zarr.codecs.ZstdCodec(level=1),
        )

        return path

    def write_append(self, file_path: Path):
        data = self.data
        store = zarr.storage.LocalStore(file_path)
        root = zarr.open_group(store=store, mode='r+')

        time_arr = root['time']
        old_len = time_arr.shape[0]
        new_time_int = (data['extra_time_data'] - np.datetime64('1970-01-01', 'D')).astype('int64')
        new_len = old_len + len(new_time_int)

        time_arr.resize(new_len)
        time_arr[old_len:new_len] = new_time_int

        temp_arr = root['temperature']
        temp_arr.resize((temp_arr.shape[0], temp_arr.shape[1], new_len))
        temp_arr[:, :, old_len:new_len] = data['extra_var_data']

    def read_full(self, file_path: Path) -> np.ndarray:
        store = zarr.storage.LocalStore(file_path)
        root = zarr.open_group(store=store, mode='r')
        return root['temperature'][:]

    def read_slice_aligned(self, file_path: Path) -> np.ndarray:
        sel = self.data['aligned_sel']
        store = zarr.storage.LocalStore(file_path)
        root = zarr.open_group(store=store, mode='r')
        return root['temperature'][sel]

    def read_slice_unaligned(self, file_path: Path) -> np.ndarray:
        sel = self.data['unaligned_sel']
        store = zarr.storage.LocalStore(file_path)
        root = zarr.open_group(store=store, mode='r')
        return root['temperature'][sel]

    def iterate_chunks(self, file_path: Path) -> int:
        total = 0
        chunk_shape = self.data['chunk_shape']
        store = zarr.storage.LocalStore(file_path)
        root = zarr.open_group(store=store, mode='r')
        temp = root['temperature']
        shape = temp.shape
        for i in range(0, shape[0], chunk_shape[0]):
            for j in range(0, shape[1], chunk_shape[1]):
                for k in range(0, shape[2], chunk_shape[2]):
                    chunk = temp[i:i + chunk_shape[0], j:j + chunk_shape[1], k:k + chunk_shape[2]]
                    total += chunk.size
        return total

    def iterate_rechunked(self, file_path: Path) -> int:
        total = 0
        store = zarr.storage.LocalStore(file_path)
        root = zarr.open_group(store=store, mode='r')
        temp = root['temperature']
        time_len = temp.shape[2]
        for k in range(0, time_len, 7):
            chunk = temp[:, :, k:k + 7]
            total += chunk.size
        return total

    def groupby_7day(self, file_path: Path) -> int:
        count = 0
        store = zarr.storage.LocalStore(file_path)
        root = zarr.open_group(store=store, mode='r')
        temp = root['temperature']
        time_len = temp.shape[2]
        for k in range(0, time_len, 7):
            chunk = temp[:, :, k:k + 7]
            chunk.mean()
            count += 1
        return count

    def get_file_size(self, file_path: Path) -> int:
        total = 0
        for f in file_path.rglob('*'):
            if f.is_file():
                total += f.stat().st_size
        return total

    def cleanup(self, file_path: Path):
        if file_path.exists():
            shutil.rmtree(file_path)
