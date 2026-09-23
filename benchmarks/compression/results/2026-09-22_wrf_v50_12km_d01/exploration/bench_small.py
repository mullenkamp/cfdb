import json, time
import numpy as np
from bench import CODECS

VARS = ['air_temperature', 'surface_pressure', 'precipitation', 'ivt']
SHAPES = {'4.3MB (24,1,324,277)': None, '180KB (1,1,324,277)': (1, 1, 324, 277),
          '49KB (24,1,32,32)': (24, 1, 32, 32), '2KB (1,1,32,32)': (1, 1, 32, 32)}
NAMES = ['zstd-1 (current)', 'lz4-1', 'blosc2 zstd-1 shuffle', 'blosc2 zstd-1 shuffle+bytedelta',
         'blosc2 lz4 shuffle', 'pcodec L4', 'pcodec L8 (default)']
MAX_BYTES = 60e6  # cap raw bytes per (var, shape) so tiny blocks don't take forever

def blocks(a, shp):
    if shp is None:
        yield a; return
    t, _, y, x = shp
    for i in range(0, a.shape[0], t):
        for j in range(0, a.shape[2], y):
            for k in range(0, a.shape[3], x):
                b = a[i:i+t, :, j:j+y, k:k+x]
                if b.shape == shp:
                    yield np.ascontiguousarray(b)

out = {}
for sname, shp in SHAPES.items():
    print(f'\n### block {sname}')
    print(f'{"codec":36s} {"ratio":>6s} {"comp MB/s":>10s} {"decomp MB/s":>12s} {"us/call comp":>13s} {"us/call dec":>12s}')
    for cname in NAMES:
        comp, decomp = CODECS[cname]
        raw = cb = ct = dt = n = 0
        for v in VARS:
            stack = np.load(f'chunks/{v}.npy')
            vb = 0
            for a in stack:
                for b in blocks(a, shp):
                    if vb > MAX_BYTES: break
                    t0 = time.perf_counter(); c = comp(b); ct += time.perf_counter() - t0
                    t0 = time.perf_counter(); o = decomp(c, str(b.dtype), b.shape); dt += time.perf_counter() - t0
                    assert np.array_equal(o, b)
                    raw += b.nbytes; cb += len(c); vb += b.nbytes; n += 1
        out[(sname, cname)] = (raw / cb, raw / ct / 1e6, raw / dt / 1e6)
        print(f'{cname:36s} {raw/cb:6.2f} {raw/ct/1e6:10.0f} {raw/dt/1e6:12.0f} {ct/n*1e6:13.1f} {dt/n*1e6:12.1f}', flush=True)
