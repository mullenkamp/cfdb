"""CF conventions multi-dimensional array database on top of Booklet"""
from cfdb.main import open_dataset, open_edataset
from cfdb.utils import compute_scale_and_offset
from rechunkit import guess_chunk_shape

__version__ = '0.1.0'
