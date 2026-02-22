"""CF conventions multi-dimensional array database on top of Booklet"""
from cfdb.main import open_dataset
from cfdb.utils import compute_scale_and_offset
from cfdb.tools import cfdb_to_netcdf4
from cfdb.combine import combine
from cfdb import dtypes
from rechunkit import guess_chunk_shape

try:
    from cfdb.edataset import open_edataset
except ImportError:
    pass

__version__ = '0.4.0'
