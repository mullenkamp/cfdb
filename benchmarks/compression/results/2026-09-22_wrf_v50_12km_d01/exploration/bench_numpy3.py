import json, time
import numpy as np, zstandard as zstd
from bench_numpy import planes_split, planes_join, hard, meta
from bench import CODECS

def elem_delta_y_shuffle():
    """wrapping element delta along y (axis -2, row-minus-row: SIMD-able), then bit-op shuffle"""
    c = zstd.ZstdCompressor(level=1); d = zstd.ZstdDecompressor()
    def comp(a):
        if a.dtype.kind != 'f':
            a = np.diff(a, axis=-2, prepend=np.zeros_like(a[..., :1, :]))
        return c.compress(b''.join(p.tobytes() for p in planes_split(a)))
    def decomp(b, dt, sh):
        raw = d.decompress(b); n = np.dtype(dt).itemsize; m = len(raw) // n
        a = planes_join([np.frombuffer(raw, 'u1', count=m, offset=i * m) for i in range(n)], dt, sh)
        if a.dtype.kind != 'f':
            a = np.cumsum(a, axis=-2, dtype=a.dtype)
        return a
    return comp, decomp

def plane_delta_y_shuffle():
    """bit-op shuffle, then wrapping uint8 delta of each byte plane along y (works for floats too)"""
    c = zstd.ZstdCompressor(level=1); d = zstd.ZstdDecompressor()
    def comp(a):
        sh2 = (-1, a.shape[-2], a.shape[-1])
        ps = [np.diff(p.reshape(sh2), axis=1, prepend=np.zeros((sh2[0] if sh2[0] > 0 else 1, 1, sh2[2]), 'u1')[:0] if False else np.zeros_like(p.reshape(sh2)[:, :1, :])) for p in planes_split(a)]
        return c.compress(b''.join(p.tobytes() for p in ps))
    def decomp(b, dt, sh):
        raw = d.decompress(b); n = np.dtype(dt).itemsize; m = len(raw) // n
        sh2 = (-1, sh[-2], sh[-1])
        ps = [np.cumsum(np.frombuffer(raw, 'u1', count=m, offset=i * m).reshape(sh2), axis=1, dtype='u1').reshape(-1) for i in range(n)]
        return planes_join(ps, dt, sh)
    return comp, decomp

C = {
    'numpy bitop-shuffle + zstd-1': __import__('bench_numpy').C['numpy bitop-shuffle + zstd-1'],
    'numpy elem-delta(y) + bitop-shuffle + zstd-1': elem_delta_y_shuffle(),
    'numpy bitop-shuffle + plane-delta(y) + zstd-1': plane_delta_y_shuffle(),
    'blosc2 zstd-1 shuffle+bytedelta (ref)': CODECS['blosc2 zstd-1 shuffle+bytedelta'],
}
for cname, (comp, decomp) in C.items():
    raw = cb = ct = dt = 0; cba = 0
    for v in meta:
        for a in np.load(f'chunks/{v}.npy'):
            bc = bd = 1e9
            for _ in range(2):
                t0 = time.perf_counter(); b = comp(a); bc = min(bc, time.perf_counter() - t0)
                t0 = time.perf_counter(); o = decomp(b, str(a.dtype), a.shape); bd = min(bd, time.perf_counter() - t0)
            assert o.dtype == a.dtype and np.array_equal(o, a), (cname, v)
            cba += len(b)
            if v in hard: raw += a.nbytes; cb += len(b); ct += bc; dt += bd
    print(f'{cname:48s} HARD ratio={raw/cb:5.2f} size={cb/1e6:6.1f} MB comp={raw/ct/1e6:5.0f} MB/s decomp={raw/dt/1e6:5.0f} MB/s | ALL file={cba/1e6:6.1f} MB', flush=True)
