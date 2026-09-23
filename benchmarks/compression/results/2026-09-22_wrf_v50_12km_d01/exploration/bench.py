"""Benchmark candidate codecs on the exact encoded chunk arrays cfdb hands to its compressor.

Every codec must round-trip bit-exactly (asserted per chunk). Timings: min of REPS runs,
single-threaded unless the codec name says otherwise. Throughput is MB of RAW (uncompressed
encoded) bytes per second, which is what matters for cfdb's per-chunk read/write path.
"""
import json
import sys
import time
import numpy as np
import zstandard as zstd
import lz4.frame
import blosc2
import bitshuffle
import zfpy
import fpzip
from pcodec import standalone, ChunkConfig, DeltaSpec, ModeSpec

REPS = 2
meta = json.load(open('chunks/meta.json'))
only = sys.argv[1:]  # optional codec-name filters

# ---------------------------------------------------------------- codec definitions
def zstd_codec(level):
    c = zstd.ZstdCompressor(level=level); d = zstd.ZstdDecompressor()
    return (lambda a: c.compress(a.tobytes()),
            lambda b, dt, sh: np.frombuffer(d.decompress(b), dt).reshape(sh))

def lz4_codec(level):
    return (lambda a: lz4.frame.compress(a.tobytes(), compression_level=level),
            lambda b, dt, sh: np.frombuffer(lz4.frame.decompress(b), dt).reshape(sh))

def np_shuffle(a):
    return np.ascontiguousarray(a.reshape(-1).view('u1').reshape(-1, a.itemsize).T).tobytes()

def np_unshuffle(b, dt, sh):
    dt = np.dtype(dt)
    return np.ascontiguousarray(np.frombuffer(b, 'u1').reshape(dt.itemsize, -1).T).reshape(-1).view(dt).reshape(sh)

def npshuf_zstd(level):
    c = zstd.ZstdCompressor(level=level); d = zstd.ZstdDecompressor()
    return (lambda a: c.compress(np_shuffle(a)),
            lambda b, dt, sh: np_unshuffle(d.decompress(b), dt, sh))

def np_delta(a):  # wrapping delta along the last (x) axis, invertible by wrapping cumsum (ints only)
    if a.dtype.kind == 'f':
        return a
    return np.diff(a, axis=-1, prepend=np.zeros_like(a[..., :1]))

def np_undelta(a):
    if a.dtype.kind == 'f':
        return a
    return np.cumsum(a, axis=-1, dtype=a.dtype)

def npdelta_shuf_zstd(level):
    c = zstd.ZstdCompressor(level=level); d = zstd.ZstdDecompressor()
    return (lambda a: c.compress(np_shuffle(np_delta(a))),
            lambda b, dt, sh: np_undelta(np_unshuffle(d.decompress(b), dt, sh)))

def blosc_codec(codec, level, filters, nthreads=1):
    def comp(a):
        return blosc2.compress2(a, cparams=dict(codec=codec, clevel=level, filters=filters,
                                                typesize=a.itemsize, nthreads=nthreads))
    def decomp(b, dt, sh):
        return np.frombuffer(blosc2.decompress2(b, dparams=dict(nthreads=nthreads)), dt).reshape(sh)
    return comp, decomp

def bitshuf_lz4():
    return (lambda a: bitshuffle.compress_lz4(np.ascontiguousarray(a)).tobytes(),
            lambda b, dt, sh: bitshuffle.decompress_lz4(np.frombuffer(b, 'u1'), sh, np.dtype(dt)))

def bitshuf_zstd(level):
    return (lambda a: bitshuffle.compress_zstd(np.ascontiguousarray(a), comp_lvl=level).tobytes(),
            lambda b, dt, sh: bitshuffle.decompress_zstd(np.frombuffer(b, 'u1'), sh, np.dtype(dt)))

def pco_codec(level, delta=None):
    kw = dict(compression_level=level, enable_8_bit=True)
    if delta is not None:
        kw['delta_spec'] = delta
    cfg = ChunkConfig(**kw)
    return (lambda a: standalone.simple_compress(np.ascontiguousarray(a).reshape(-1), cfg),
            lambda b, dt, sh: standalone.simple_decompress(b).reshape(sh))

_ZFP_CAST = {'uint16': 'int32', 'uint8': 'int32', 'uint32': 'int64', 'float32': 'float32', 'int32': 'int32'}
def zfp_lossless():
    def comp(a):
        a3 = a.reshape([s for s in a.shape if s != 1]).astype(_ZFP_CAST[str(a.dtype)])
        return zfpy.compress_numpy(a3)
    def decomp(b, dt, sh):
        return zfpy.decompress_numpy(b).astype(dt).reshape(sh)
    return comp, decomp

def fpzip_lossless():
    return (lambda a: fpzip.compress(np.ascontiguousarray(a.reshape([s for s in a.shape if s != 1]))),
            lambda b, dt, sh: fpzip.decompress(b).reshape(sh))

F = blosc2.Filter; C = blosc2.Codec
CODECS = {
    # --- baselines (what cfdb offers today)
    'zstd-1 (current)':        zstd_codec(1),
    'zstd-3':                  zstd_codec(3),
    'zstd-6':                  zstd_codec(6),
    'zstd-9':                  zstd_codec(9),
    'lz4-1':                   lz4_codec(1),
    # --- pure-numpy filters in front of zstd (no new dependency)
    'npshuffle+zstd-1':        npshuf_zstd(1),
    'npshuffle+zstd-3':        npshuf_zstd(3),
    'npdelta+npshuffle+zstd-1': npdelta_shuf_zstd(1),
    'npdelta+npshuffle+zstd-3': npdelta_shuf_zstd(3),
    # --- blosc2 (C-level filters, blocked)
    'blosc2 lz4 shuffle':      blosc_codec(C.LZ4, 1, [F.SHUFFLE]),
    'blosc2 lz4 bitshuffle':   blosc_codec(C.LZ4, 1, [F.BITSHUFFLE]),
    'blosc2 zstd-1 shuffle':   blosc_codec(C.ZSTD, 1, [F.SHUFFLE]),
    'blosc2 zstd-1 bitshuffle': blosc_codec(C.ZSTD, 1, [F.BITSHUFFLE]),
    'blosc2 zstd-1 shuffle+bytedelta': blosc_codec(C.ZSTD, 1, [F.SHUFFLE, F.BYTEDELTA]),
    'blosc2 zstd-3 shuffle+bytedelta': blosc_codec(C.ZSTD, 3, [F.SHUFFLE, F.BYTEDELTA]),
    'blosc2 zstd-5 shuffle+bytedelta': blosc_codec(C.ZSTD, 5, [F.SHUFFLE, F.BYTEDELTA]),
    'blosc2 zstd-9 shuffle+bytedelta': blosc_codec(C.ZSTD, 9, [F.SHUFFLE, F.BYTEDELTA]),
    'blosc2 zstd-1 bitshuffle+bytedelta': blosc_codec(C.ZSTD, 1, [F.BITSHUFFLE, F.BYTEDELTA]),
    'blosc2 lz4 shuffle+bytedelta': blosc_codec(C.LZ4, 1, [F.SHUFFLE, F.BYTEDELTA]),
    'blosc2 blosclz shuffle':  blosc_codec(C.BLOSCLZ, 5, [F.SHUFFLE]),
    'blosc2 zstd-1 shuffle+bytedelta 12thr': blosc_codec(C.ZSTD, 1, [F.SHUFFLE, F.BYTEDELTA], nthreads=12),
    # --- bitshuffle package (used by HDF5 ecosystem)
    'bitshuffle+lz4':          bitshuf_lz4(),
    'bitshuffle+zstd-1':       bitshuf_zstd(1),
    # --- numeric-specialised codecs
    'pcodec L2':               pco_codec(2),
    'pcodec L4':               pco_codec(4),
    'pcodec L8 (default)':     pco_codec(8),
    'pcodec L12':              pco_codec(12),
    'pcodec L8 delta1':        pco_codec(8, DeltaSpec.try_consecutive(1)),
    'pcodec L8 lookback':      pco_codec(8, DeltaSpec.try_lookback()),
    'zfp lossless':            zfp_lossless(),
    'fpzip lossless (f32 only)': fpzip_lossless(),
}
FLOAT_ONLY = {'fpzip lossless (f32 only)'}

# ---------------------------------------------------------------- run
if __name__ == '__main__':
    results = {}
    names = [n for n in CODECS if not only or any(o in n for o in only)]
    for cname in names:
        comp, decomp = CODECS[cname]
        results[cname] = {}
        t_all = time.perf_counter()
        for var, m in meta.items():
            dt = m['enc_dtype']
            if cname in FLOAT_ONLY and not dt.startswith('float'):
                continue
            stack = np.load(f'chunks/{var}.npy')
            rec = dict(raw=0, comp=0, ct=0.0, dt=0.0)
            for a in stack:
                best_c = best_d = 1e9
                for _ in range(REPS):
                    t0 = time.perf_counter(); b = comp(a); best_c = min(best_c, time.perf_counter() - t0)
                for _ in range(REPS):
                    t0 = time.perf_counter(); out = decomp(b, dt, a.shape); best_d = min(best_d, time.perf_counter() - t0)
                assert out.dtype == a.dtype and out.shape == a.shape and np.array_equal(out, a), (cname, var)
                rec['raw'] += a.nbytes; rec['comp'] += len(b); rec['ct'] += best_c; rec['dt'] += best_d
            results[cname][var] = rec
        r = sum(v['raw'] for v in results[cname].values()); c = sum(v['comp'] for v in results[cname].values())
        ct = sum(v['ct'] for v in results[cname].values()); dtt = sum(v['dt'] for v in results[cname].values())
        print(f'{cname:42s} ratio={r/c:6.2f}  comp={r/ct/1e6:7.0f} MB/s  decomp={r/dtt/1e6:7.0f} MB/s  ({time.perf_counter()-t_all:.0f}s)', flush=True)
        json.dump(results, open('results.json' if not only else 'results_partial.json', 'w'), indent=1)
