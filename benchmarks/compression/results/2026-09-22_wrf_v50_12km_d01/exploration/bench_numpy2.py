import json, time
import numpy as np, zstandard as zstd
from bench_numpy import planes_split, planes_join, hard, meta

def variant(delta_planes, cumsum_dtype='u1'):
    """delta_planes: set of plane indices (0 = low byte) to delta; cumsum via wrapping uint8 or via int32 then cast"""
    c = zstd.ZstdCompressor(level=1); d = zstd.ZstdDecompressor()
    def comp(a):
        ps = planes_split(a)
        ps = [np.diff(p, prepend=np.uint8(0)) if i in delta_planes else p for i, p in enumerate(ps)]
        return c.compress(b''.join(p.tobytes() for p in ps))
    def decomp(b, dt, sh):
        raw = d.decompress(b); n = np.dtype(dt).itemsize; m = len(raw) // n
        ps = [np.frombuffer(raw, 'u1', count=m, offset=i * m) for i in range(n)]
        if cumsum_dtype == 'u1':
            ps = [np.cumsum(p, dtype='u1') if i in delta_planes else p for i, p in enumerate(ps)]
        else:
            ps = [np.cumsum(p, dtype=cumsum_dtype).astype('u1') if i in delta_planes else p for i, p in enumerate(ps)]
        return planes_join(ps, dt, sh)
    return comp, decomp

C = {
    'shuffle, delta all planes, cumsum u1': variant({0,1,2,3}),
    'shuffle, delta all planes, cumsum i8->u1': variant({0,1,2,3}, 'i8'),
    'shuffle, delta planes>=1 (skip low byte)': variant({1,2,3}),
    'shuffle, delta top plane only (u16: byte1; u32: byte3)': None,  # filled below per dtype
}
def top_only():
    c = zstd.ZstdCompressor(level=1); d = zstd.ZstdDecompressor()
    def comp(a):
        ps = planes_split(a); k = len(ps) - 1
        if k: ps[k] = np.diff(ps[k], prepend=np.uint8(0))
        return c.compress(b''.join(p.tobytes() for p in ps))
    def decomp(b, dt, sh):
        raw = d.decompress(b); n = np.dtype(dt).itemsize; m = len(raw) // n
        ps = [np.frombuffer(raw, 'u1', count=m, offset=i * m) for i in range(n)]
        if n > 1: ps[n-1] = np.cumsum(ps[n-1], dtype='u1')
        return planes_join(ps, dt, sh)
    return comp, decomp
C['shuffle, delta top plane only (u16: byte1; u32: byte3)'] = top_only()

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
    print(f'{cname:52s} HARD ratio={raw/cb:5.2f} size={cb/1e6:6.1f} MB comp={raw/ct/1e6:5.0f} MB/s decomp={raw/dt/1e6:5.0f} MB/s | ALL file={cba/1e6:6.1f} MB', flush=True)
