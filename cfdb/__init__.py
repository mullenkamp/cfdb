"""CF conventions multi-dimensional array database on top of Booklet"""
from cfdb.main import open_dataset
from cfdb.data_models import PartialDataWarning
from cfdb.utils import compute_scale_and_offset
from cfdb.tools import cfdb_to_netcdf4
from cfdb.combine import combine
from cfdb.merge import merge_into
from cfdb import dtypes
from rechunkit import guess_chunk_shape

try:
    from cfdb.edataset import open_edataset
except (ImportError, AttributeError):
    pass

__version__ = '0.9.2'
