"""CF conventions multi-dimensional array database on top of Booklet"""
from cfdb.main import open_dataset, open_edataset
from cfdb.utils import compute_scale_and_offset
from cfdb.tools import netcdf4_to_cfdb, cfdb_to_netcdf4
from cfdb import dtypes
from rechunkit import guess_chunk_shape

__version__ = '0.1.5'
