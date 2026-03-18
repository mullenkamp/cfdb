from pathlib import Path

import numpy as np

from cfdb import open_dataset, dtypes
from benchmarks.backends.base import BackendBase


class CfdbBackend(BackendBase):
    name = 'cfdb'

    def setup(self, base_dir: Path, data_config: dict):
        self.base_dir = base_dir
        self.data = data_config
        self._counter = 0

    def _next_path(self):
        self._counter += 1
        return self.base_dir / f'bench_{self._counter}.cfdb'

    def write_full(self) -> Path:
        path = self._next_path()
        data = self.data
        data_dtype = dtypes.dtype('float32')

        with open_dataset(path, flag='n') as ds:
            ds.create.coord.lat(data=data['lat_data'], chunk_shape=(data['chunk_shape'][0],))
            ds.create.coord.lon(data=data['lon_data'], chunk_shape=(data['chunk_shape'][1],))
            ds.create.coord.time(data=data['time_data'], dtype=data['time_data'].dtype)

            dv = ds.create.data_var.generic(
                'temperature',
                ('latitude', 'longitude', 'time'),
                data_dtype,
                chunk_shape=data['chunk_shape'],
            )
            dv[:] = data['var_data']

        return path

    def write_append(self, file_path: Path):
        data = self.data
        with open_dataset(file_path, flag='w') as ds:
            time_coord = ds['time']
            old_len = len(time_coord.data)
            time_coord.append(data['extra_time_data'])

            dv = ds['temperature']
            new_len = old_len + len(data['extra_time_data'])
            dv[slice(None), slice(None), slice(old_len, new_len)] = data['extra_var_data']

    def read_full(self, file_path: Path) -> np.ndarray:
        with open_dataset(file_path) as ds:
            return ds['temperature'].data

    def read_slice_aligned(self, file_path: Path) -> np.ndarray:
        sel = self.data['aligned_sel']
        with open_dataset(file_path) as ds:
            return ds['temperature'][sel].data

    def read_slice_unaligned(self, file_path: Path) -> np.ndarray:
        sel = self.data['unaligned_sel']
        with open_dataset(file_path) as ds:
            return ds['temperature'][sel].data

    def iterate_chunks(self, file_path: Path) -> int:
        total = 0
        with open_dataset(file_path) as ds:
            for _, chunk_data in ds['temperature'].iter_chunks():
                total += chunk_data.size
        return total

    def iterate_rechunked(self, file_path: Path) -> int:
        total = 0
        rechunked = self.data['rechunked_shape']
        with open_dataset(file_path) as ds:
            for _, chunk_data in ds['temperature'].iter_chunks(rechunked, max_mem=2**31):
                total += chunk_data.size
        return total

    def groupby_7day(self, file_path: Path) -> int:
        count = 0
        with open_dataset(file_path) as ds:
            for _, chunk_data in ds['temperature'].groupby({'time': '7D'}, max_mem=2**31):
                chunk_data.mean()
                count += 1
        return count

    def get_file_size(self, file_path: Path) -> int:
        return file_path.stat().st_size

    def cleanup(self, file_path: Path):
        if file_path.exists():
            file_path.unlink()
