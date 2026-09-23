"""Candidate chunk codecs for cfdb, all exposed as ``(compress, decompress)`` pairs.

``compress(arr) -> bytes`` takes the ENCODED chunk array exactly as cfdb hands it to its
compressor (packed uint16/uint32, raw float32, ...). ``decompress(buf, dtype, shape) -> arr``
must return it bit-identically; the benchmarks assert this on every chunk.

Core codecs need only cfdb's own dependencies (numpy, zstandard, lz4). The optional groups
are registered only when their package imports: ``blosc``, ``blosc2``, ``pcodec``, ``zfpy``.
"""
import numpy as np
import zstandard as zstd
import lz4.frame


# ----------------------------------------------------------------------------- filters
def byte_planes(a):
    """Split an array into its little-endian byte planes (plane 0 = low byte), each uint8.

    Bit-op formulation: ``astype('u1')`` truncates, so ``(u >> 8*i).astype('u1')`` is plane i.
    This vectorises; a ``view('u1').reshape(-1, itemsize).T`` transpose is ~3x slower to undo.
    """
    n = a.dtype.itemsize
    flat = a.reshape(-1)
    if n == 1:
        return [flat.view('u1')]
    u = flat.view(f'u{n}')
    return [u.astype('u1')] + [(u >> (8 * i)).astype('u1') for i in range(1, n)]


def join_planes(planes, dtype, shape):
    dtype = np.dtype(dtype)
    n = dtype.itemsize
    if n == 1:
        return planes[0].view(dtype).reshape(shape)
    u = planes[-1].astype(f'u{n}')
    for i in range(n - 2, -1, -1):
        u <<= 8
        u |= planes[i]
    return u.view(dtype).reshape(shape)


def split_planes_from_bytes(buf, dtype):
    n = np.dtype(dtype).itemsize
    m = len(buf) // n
    return [np.frombuffer(buf, 'u1', count=m, offset=i * m) for i in range(n)]


def delta_axis(shape):
    """Axis for the y-delta filter: the second-to-last axis when it has extent > 1, else None."""
    if len(shape) >= 2 and shape[-2] > 1:
        return len(shape) - 2
    return None


# ----------------------------------------------------------------------------- codecs
def zstd_codec(level):
    c = zstd.ZstdCompressor(level=level)
    d = zstd.ZstdDecompressor()
    return (lambda a: c.compress(a.tobytes()),
            lambda b, dt, sh: np.frombuffer(d.decompress(b), dt).reshape(sh))


def lz4_codec(level):
    return (lambda a: lz4.frame.compress(a.tobytes(), compression_level=level),
            lambda b, dt, sh: np.frombuffer(lz4.frame.decompress(b), dt).reshape(sh))


def shuffle_zstd(level):
    """numpy byte-shuffle then zstd — no dependency beyond cfdb's own."""
    c = zstd.ZstdCompressor(level=level)
    d = zstd.ZstdDecompressor()
    return (lambda a: c.compress(b''.join(p.tobytes() for p in byte_planes(a))),
            lambda b, dt, sh: join_planes(split_planes_from_bytes(d.decompress(b), dt), dt, sh))


def shuffle_lz4(level):
    """numpy byte-shuffle then lz4 frame — the lz4 analogue of shuffle_zstd."""
    return (lambda a: lz4.frame.compress(b''.join(p.tobytes() for p in byte_planes(a)), compression_level=level),
            lambda b, dt, sh: join_planes(split_planes_from_bytes(lz4.frame.decompress(b), dt), dt, sh))


def shuffle_ydelta_zstd(level):
    """numpy byte-shuffle, then a wrapping uint8 delta of every byte plane along the
    second-to-last chunk axis (row-minus-row, so both directions vectorise). The inverse is
    a loop of ``shape[-2]`` slab adds, so decode cost grows with that extent."""
    c = zstd.ZstdCompressor(level=level)
    d = zstd.ZstdDecompressor()

    def comp(a):
        ax = delta_axis(a.shape)
        planes = byte_planes(a)
        if ax is not None:
            planes = [np.diff(p.reshape(a.shape), axis=ax,
                              prepend=np.zeros_like(np.take(p.reshape(a.shape), [0], axis=ax)))
                      for p in planes]
        return c.compress(b''.join(p.tobytes() for p in planes))

    def decomp(b, dt, sh):
        ax = delta_axis(sh)
        planes = split_planes_from_bytes(d.decompress(b), dt)
        if ax is not None:
            out = []
            for plane in planes:
                p = plane.reshape(sh).copy()
                pm = np.moveaxis(p, ax, 0)
                for j in range(1, sh[ax]):
                    pm[j] += pm[j - 1]
                out.append(p.reshape(-1))
            planes = out
        return join_planes(planes, dt, sh)

    return comp, decomp


CODECS = {
    'zstd-1': zstd_codec(1),
    'zstd-3': zstd_codec(3),
    'lz4-1': lz4_codec(1),
    'shuffle+zstd-1': shuffle_zstd(1),
    'shuffle+zstd-3': shuffle_zstd(3),
    'shuffle+lz4-1': shuffle_lz4(1),
    'shuffle+ydelta+zstd-1': shuffle_ydelta_zstd(1),
}
DEFAULT = ['zstd-1', 'shuffle+zstd-1', 'shuffle+ydelta+zstd-1']

# ------------------------------------------------------------------- optional codecs
try:
    import blosc  # c-blosc1, zero-dependency wheel

    def blosc1_codec(shuffle, level=1, cname='zstd'):
        return (lambda a: blosc.compress(a.tobytes(), typesize=a.itemsize, clevel=level, shuffle=shuffle, cname=cname),
                lambda b, dt, sh: np.frombuffer(blosc.decompress(b), dt).reshape(sh))

    blosc.set_nthreads(1)
    CODECS['blosc1 shuffle zstd-1'] = blosc1_codec(blosc.SHUFFLE)
    CODECS['blosc1 bitshuffle zstd-1'] = blosc1_codec(blosc.BITSHUFFLE)
except ImportError:
    pass

try:
    import blosc2

    def blosc2_codec(codec, level, filters, nthreads=1):
        def comp(a):
            return blosc2.compress2(a, cparams=dict(codec=codec, clevel=level, filters=filters,
                                                    typesize=a.itemsize, nthreads=nthreads))

        def decomp(b, dt, sh):
            return np.frombuffer(blosc2.decompress2(b, dparams=dict(nthreads=nthreads)), dt).reshape(sh)

        return comp, decomp

    _F, _C = blosc2.Filter, blosc2.Codec
    CODECS['blosc2 shuffle zstd-1'] = blosc2_codec(_C.ZSTD, 1, [_F.SHUFFLE])
    CODECS['blosc2 shuffle+bytedelta zstd-1'] = blosc2_codec(_C.ZSTD, 1, [_F.SHUFFLE, _F.BYTEDELTA])
    CODECS['blosc2 shuffle lz4'] = blosc2_codec(_C.LZ4, 1, [_F.SHUFFLE])
except ImportError:
    pass

try:
    from pcodec import standalone as _pco, ChunkConfig as _PcoConfig

    def pcodec_codec(level):
        cfg = _PcoConfig(compression_level=level, enable_8_bit=True)
        return (lambda a: _pco.simple_compress(np.ascontiguousarray(a).reshape(-1), cfg),
                lambda b, dt, sh: _pco.simple_decompress(b).reshape(sh))

    CODECS['pcodec L4'] = pcodec_codec(4)
    CODECS['pcodec L8'] = pcodec_codec(8)
except ImportError:
    pass

try:
    import zfpy

    _ZFP_CAST = {'uint8': 'int32', 'uint16': 'int32', 'int16': 'int32', 'uint32': 'int64',
                 'int32': 'int32', 'int64': 'int64', 'float32': 'float32', 'float64': 'float64'}

    def zfp_lossless():
        def comp(a):
            a = a.reshape([s for s in a.shape if s != 1] or [1]).astype(_ZFP_CAST[str(a.dtype)])
            return zfpy.compress_numpy(a)
        return comp, lambda b, dt, sh: zfpy.decompress_numpy(b).astype(dt).reshape(sh)

    CODECS['zfp lossless'] = zfp_lossless()
except ImportError:
    pass
