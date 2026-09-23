"""Stream the stored chunks of a cfdb dataset as the ENCODED arrays its compressor sees.

Uses the private Booklet handle (``ds._blt``) so the stored (compressed) size of each chunk
is available alongside the array; nothing is held beyond one chunk at a time.
"""
import numpy as np
import cfdb


def encoded_dtype(var):
    dt = var.dtype
    return np.dtype(dt.dtype_encoded if dt.dtype_encoded is not None else dt.dtype_decoded)


def is_fixed_width(var):
    """String and geometry variables are not fixed-width arrays; ``frombuffer`` cannot read them."""
    return encoded_dtype(var).kind not in 'OUT'


def _chunk_start(key):
    return tuple(int(x) for x in key.split('!', 1)[1].split('.'))


def iter_var_chunks(ds, var_name):
    """Yield ``(encoded_array, stored_bytes)`` for every stored chunk of one variable, in
    numeric chunk-start order (Booklet keys are ``name!t.z.y.x`` with no zero padding, so a
    plain string sort puts ``120`` before ``24``). Only full-shape chunks exist: every cfdb
    write path stores a full ``chunk_shape`` array."""
    var = ds[var_name]
    dtype = encoded_dtype(var)
    prefix = f'{var_name}!'
    for key in sorted((k for k in ds._blt.keys() if k.startswith(prefix)), key=_chunk_start):
        buf = ds._blt.get(key)
        raw = var.compressor.decompress(buf)
        arr = np.frombuffer(raw, dtype=dtype).reshape(var.chunk_shape)
        yield arr, len(buf)


def var_info(ds, var_name):
    var = ds[var_name]
    dt = var.dtype
    return dict(enc_dtype=str(encoded_dtype(var)), dec_dtype=str(dt.dtype_decoded), chunk_shape=list(var.chunk_shape),
                shape=list(var.shape), precision=dt.precision, offset=dt.offset, fillvalue=getattr(dt, 'fillvalue', None))


def open_ro(path):
    return cfdb.open_dataset(path)
