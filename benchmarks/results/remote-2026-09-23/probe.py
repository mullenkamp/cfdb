"""Remote read cost vs request size on a public, UNGROUPED EDataset (one HTTP object per chunk)."""
import time, statistics, json
from concurrent.futures import ThreadPoolExecutor
from cfdb import open_edataset
URL = 'https://b2.tethys-ts.xyz/file/ecmwf-forecasts/ifs/nz_0p25.cfdb'
S = '/tmp/claude-1000/-home-mike-git-cfdb-repos-cfdb/70ccb4ba-b8ea-45c1-b2d0-d12f421c0efa/scratchpad/remote'
ds = open_edataset(URL, f'{S}/probe.cfdb')
sess = ds._blt._remote_session
keys = sorted(k for k in ds._blt._remote_index.keys() if k.startswith('air_temperature_2m!'))
def get(key, a=None, b=None):
    t0 = time.perf_counter(); r = sess.get_object(key, range_start=a, range_end=b); t = time.perf_counter() - t0
    assert r.status in (200, 206), (r.status, key)
    return t, len(r.data)
out = {}
# 1. full objects, sequential: size + time
seq = [get(k) for k in keys[:8]]
full = seq[0][1]
print('full chunk objects (compressed bytes, seconds):', [(n, round(t, 3)) for t, n in seq])
# 2. byte ranges of ONE object, sequential, 5 reps each
k0 = keys[0]
print(f'\nranged GETs on {k0} (object {full/1e6:.2f} MB):')
print(f'{"bytes":>9s} {"min s":>7s} {"median s":>9s} {"MB/s (median)":>14s}')
rng = []
for n in [1_000, 16_000, 64_000, 256_000, 1_000_000, full]:
    n = min(n, full)
    ts = [get(k0, 0, n - 1)[0] for _ in range(5)]
    rng.append((n, min(ts), statistics.median(ts)))
    print(f'{n:9d} {min(ts):7.3f} {statistics.median(ts):9.3f} {n/statistics.median(ts)/1e6:14.1f}')
out['ranges'] = rng
# 3. parallel: 10 threads
def par(tasks):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(10) as ex:
        res = list(ex.map(lambda a: get(*a), tasks))
    return time.perf_counter() - t0, sum(n for _, n in res)
for label, tasks in [('10 x 16 KB ranges', [(k, 0, 15_999) for k in keys[:10]]),
                     ('20 x 16 KB ranges', [(k, 0, 15_999) for k in keys[:20]]),
                     ('10 x full objects', [(k,) for k in keys[8:18]]),
                     ('20 x full objects', [(k,) for k in keys[8:28]])]:
    t, nb = par(tasks)
    print(f'parallel {label:18s}: {t:6.3f} s, {nb/1e6:7.1f} MB -> {nb/t/1e6:6.1f} MB/s aggregate, {len(tasks)/t:5.1f} requests/s')
json.dump(out, open(f'{S}/probe.json', 'w'))
ds.close()
