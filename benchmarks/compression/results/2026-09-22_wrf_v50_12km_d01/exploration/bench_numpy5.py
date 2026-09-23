import json, time
import numpy as np, zstandard as zstd
from bench_numpy import planes_split, planes_join, hard, meta

def plane_delta_axis(axis_sel):
    """bit-op shuffle, then wrapping uint8 delta of each byte plane along the chosen chunk axis;
    inverse is a loop of slab adds along that axis"""
    c = zstd.ZstdCompressor(level=1); d = zstd.ZstdDecompressor()
    def comp(a):
        ax = axis_sel(a.shape); out = []
        for p in planes_split(a):
            p = p.reshape(a.shape)
            out.append(np.diff(p, axis=ax, prepend=np.zeros_like(np.take(p, [0], axis=ax))))
        return c.compress(b''.join(p.tobytes() for p in out))
    def decomp(b, dt, sh):
        raw = d.decompress(b); n = np.dtype(dt).itemsize; m = len(raw) // n
        ax = axis_sel(sh); ps = []
        for i in range(n):
            p = np.frombuffer(raw, 'u1', count=m, offset=i * m).reshape(sh).copy()
            pm = np.moveaxis(p, ax, 0)
            for j in range(1, sh[ax]):
                pm[j] += pm[j - 1]
            ps.append(p.reshape(-1))
        return planes_join(ps, dt, sh)
    return comp, decomp

C = {
    'plane-delta along time (axis 0, 24 steps)': plane_delta_axis(lambda sh: 0),
    'plane-delta along y (axis 2, 324 steps)':   plane_delta_axis(lambda sh: 2),
    'plane-delta along x (axis 3, 277 steps)':   plane_delta_axis(lambda sh: 3),
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
