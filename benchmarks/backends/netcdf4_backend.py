import shutil
from pathlib import Path

import numpy as np
import h5netcdf

from benchmarks.backends.base import BackendBase


class NetCDF4Backend(BackendBase):
    name = 'netcdf4'

    def setup(self, base_dir: Path, data_config: dict):
        self.base_dir = base_dir
        self.data = data_config
        self._counter = 0

    def _next_path(self):
        self._counter += 1
        return self.base_dir / f'bench_{self._counter}.nc'

    def write_full(self) -> Path:
        path = self._next_path()
        data = self.data
        lat_size, lon_size, time_size = data['var_data'].shape
        chunk_shape = data['chunk_shape']

        with h5netcdf.File(path, 'w') as f:
            f.dimensions['latitude'] = lat_size
            f.dimensions['longitude'] = lon_size
            f.dimensions['time'] = None  # unlimited for append support

            f.create_variable('latitude', ('latitude',), data=data['lat_data'])
            f.create_variable('longitude', ('longitude',), data=data['lon_data'])

            # Unlimited dim: create empty then resize and write
            time_int = (data['time_data'] - np.datetime64('1970-01-01', 'D')).astype('int64')
            time_var = f.create_variable('time', ('time',), dtype='int64')
            time_var.attrs['units'] = 'days since 1970-01-01'

            f.create_variable(
                'temperature',
                ('latitude', 'longitude', 'time'),
                dtype='float32',
                compression='gzip',
                compression_opts=4,
                chunks=chunk_shape,
            )

            f.resize_dimension('time', time_size)
            time_var[:] = time_int
            f['temperature'][:] = data['var_data']

        return path

    def write_append(self, file_path: Path):
        data = self.data
        with h5netcdf.File(file_path, 'a') as f:
            time_var = f['time']
            old_len = time_var.shape[0]
            new_time_int = (data['extra_time_data'] - np.datetime64('1970-01-01', 'D')).astype('int64')
            new_len = old_len + len(new_time_int)

            f.resize_dimension('time', new_len)
            time_var[old_len:new_len] = new_time_int
            f['temperature'][..., old_len:new_len] = data['extra_var_data']

    def read_full(self, file_path: Path) -> np.ndarray:
        with h5netcdf.File(file_path, 'r') as f:
            return f['temperature'][:]

    def read_slice_aligned(self, file_path: Path) -> np.ndarray:
        sel = self.data['aligned_sel']
        with h5netcdf.File(file_path, 'r') as f:
            return f['temperature'][sel]

    def read_slice_unaligned(self, file_path: Path) -> np.ndarray:
        sel = self.data['unaligned_sel']
        with h5netcdf.File(file_path, 'r') as f:
            return f['temperature'][sel]

    def iterate_chunks(self, file_path: Path) -> int:
        total = 0
        chunk_shape = self.data['chunk_shape']
        with h5netcdf.File(file_path, 'r') as f:
            temp = f['temperature']
            shape = temp.shape
            for i in range(0, shape[0], chunk_shape[0]):
                for j in range(0, shape[1], chunk_shape[1]):
                    for k in range(0, shape[2], chunk_shape[2]):
                        chunk = temp[i:i + chunk_shape[0], j:j + chunk_shape[1], k:k + chunk_shape[2]]
                        total += chunk.size
        return total

    def iterate_rechunked(self, file_path: Path) -> int:
        total = 0
        data = self.data
        lat_size = data['lat_data'].shape[0]
        lon_size = data['lon_data'].shape[0]
        with h5netcdf.File(file_path, 'r') as f:
            temp = f['temperature']
            time_len = temp.shape[2]
            for k in range(0, time_len, 7):
                chunk = temp[:lat_size, :lon_size, k:k + 7]
                total += chunk.size
        return total

    def groupby_7day(self, file_path: Path) -> int:
        count = 0
        with h5netcdf.File(file_path, 'r') as f:
            temp = f['temperature']
            time_len = temp.shape[2]
            for k in range(0, time_len, 7):
                chunk = temp[:, :, k:k + 7]
                chunk.mean()
                count += 1
        return count

    def get_file_size(self, file_path: Path) -> int:
        return file_path.stat().st_size

    def cleanup(self, file_path: Path):
        if file_path.exists():
            file_path.unlink()
