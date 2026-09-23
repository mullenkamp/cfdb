import json, time
import numpy as np, blosc
from bench import CODECS
from bench_numpy import C as CN, hard, meta
from bench_numpy5 import plane_delta_axis
blosc.set_nthreads(1)
FIN = {
    'zstd-1 (current)': CODECS['zstd-1 (current)'],
    'np shuffle+zstd-1': CN['numpy bitop-shuffle + zstd-1'],
    'np shuffle+ydelta+zstd-1': plane_delta_axis(lambda sh: 2),
    'blosc1 shuffle zstd-1': CN['blosc1 zstd-1 shuffle (zero-dep pkg)'],
    'pcodec L8': CODECS['pcodec L8 (default)'],
}
VARS = ['air_temperature', 'surface_pressure', 'wind_speed', 'precipitation', 'mixing_ratio', 'mslp', 'ivt']
SHAPES = [(24,1,324,277), (24,1,162,277), (24,1,81,138), (24,1,40,69), (24,1,20,34), (24,1,10,17), (24,1,5,8),
          (1,1,324,277), (1,1,162,138), (1,1,81,69), (1,1,40,34), (1,1,20,17), (1,1,10,8)]
MAX_BYTES = 40e6

def blocks(a, shp):
    t, _, y, x = shp
    for i in range(0, a.shape[0], t):
        for j in range(0, a.shape[2], y):
            for k in range(0, a.shape[3], x):
                b = a[i:i+t, :, j:j+y, k:k+x]
                if b.shape == shp:
                    yield np.ascontiguousarray(b)

stacks = {v: np.load(f'chunks/{v}.npy') for v in VARS}
rows = []
for shp in SHAPES:
    nb = int(np.prod(shp))
    row = {'shape': shp}
    for cname, (comp, decomp) in FIN.items():
        raw = cb = ct = dt = n = 0
        for v in VARS:
            vb = 0
            for a in stacks[v]:
                for b in blocks(a, shp):
                    if vb > MAX_BYTES: break
                    t0 = time.perf_counter(); c = comp(b); ct += time.perf_counter() - t0
                    t0 = time.perf_counter(); o = decomp(c, str(b.dtype), b.shape); dt += time.perf_counter() - t0
                    assert np.array_equal(o, b)
                    raw += b.nbytes; cb += len(c); vb += b.nbytes; n += 1
        row[cname] = (raw / cb, raw / ct / 1e6, raw / dt / 1e6, cb / n)
    rows.append(row)
    print('.', end='', flush=True)
print()
names = list(FIN)
for k, title in [(0, 'COMPRESSION RATIO'), (1, 'COMPRESS MB/s'), (2, 'DECOMPRESS MB/s')]:
    print(f'\n=== {title} vs block shape (7 hard vars; elems x 2-4 B = raw bytes) ===')
    print(f'{"block shape":18s} {"elems":>8s} ' + ''.join(f'{n:>26s}' for n in names))
    for r in rows:
        fmt = '{:26.2f}' if k == 0 else '{:26.0f}'
        print(f'{str(r["shape"]):18s} {int(np.prod(r["shape"])):8d} ' + ''.join(fmt.format(r[n][k]) for n in names))
json.dump([{**{'shape': r['shape']}, **{n: r[n] for n in names}} for r in rows], open('results_sizes.json', 'w'), indent=1)
