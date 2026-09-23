"""Measured cfdb rechunking/selection cost vs default chunk target, on real d01 variables rewritten at
the chunk shape cfdb would guess for each byte target. Warm page cache, single thread, min of 3."""
import sys, time, math, gc, tempfile, shutil, warnings, numpy as np, rechunkit
from pathlib import Path
from cfdb import open_dataset
from benchmarks.profile_cfdb import _create_like
warnings.simplefilter('ignore')
SRC = '/home/mike/data/wrf/sst/v50_12km_wvt_8_regions/cfdb_cache/d01.cfdb'
var_name = sys.argv[1]
TARGETS = [2**18, 2**19, 2**20, 2**21, 2**22]
def best(f, n=3):
    f(); b = 1e9
    for _ in range(n):
        gc.collect(); t0 = time.perf_counter(); f(); b = min(b, time.perf_counter() - t0)
    return b
src = open_dataset(SRC); sv = src[var_name]
dt = sv.dtype; it = np.dtype(dt.dtype_encoded or dt.dtype_decoded).itemsize
blocks = [(sl, np.array(d)) for sl, d in sv.iter_chunks()]
n0 = sv.shape[0]; T = n0
tmp = Path(tempfile.mkdtemp())
ops = {
  'iter_chunks (storage)':        lambda v, mm: [None for _ in v.iter_chunks()],
  'rechunk -> pixel series':      lambda v, mm: [None for _ in v.iter_chunks(chunk_shape={'time': T, 'y': 10, 'x': 10}, max_mem=mm)],
  'rechunk -> hourly grids':      lambda v, mm: [None for _ in v.iter_chunks(chunk_shape={'time': 1}, max_mem=mm)],
  'groupby daily mean':           lambda v, mm: [d.mean() for _, d in v.groupby({'time': 'D'}, max_mem=mm)],
  'one point, all times':         lambda v, mm: v[:, 0, 150, 150].data,
  'one time step, full grid':     lambda v, mm: v[100, 0, :, :].data,
}
print(f'{var_name} {sv.shape} itemsize {it}; seconds, min of 3; max_mem 512 MiB / 4 MiB')
hdr = f'{"target":>9s} {"chunk shape":>20s} {"el":>8s} '
print(hdr + ''.join(f'{k[:22]:>24s}' for k in ops))
for t in TARGETS:
    cs = rechunkit.guess_chunk_shape(sv.shape, it, t)
    p = tmp / 'r.cfdb'
    if p.exists(): p.unlink()
    ds, dv = _create_like(src, sv, p, cs, n0)
    for sl, d in blocks: dv[sl] = d
    ds.close()
    ds = open_dataset(p); v = ds[var_name]
    cells = []
    for k, op in ops.items():
        a = best(lambda: op(v, 2**29))
        b = best(lambda: op(v, 2**22)) if k.startswith(('rechunk', 'groupby')) else None
        cells.append(f'{a:8.3f}' + (f' /{b:8.3f}' if b is not None else ' ' * 10))
    print(f'{t//1024:>6d}KiB {str(cs):>20s} {math.prod(cs):8d} ' + ''.join(f'{c:>24s}' for c in cells), flush=True)
    ds.close()
src.close(); shutil.rmtree(tmp)
