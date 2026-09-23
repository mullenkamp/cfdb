import time
from concurrent.futures import ThreadPoolExecutor
from cfdb import open_edataset
S = '/tmp/claude-1000/-home-mike-git-cfdb-repos-cfdb/70ccb4ba-b8ea-45c1-b2d0-d12f421c0efa/scratchpad/remote'
ds = open_edataset('https://b2.tethys-ts.xyz/file/ecmwf-forecasts/ifs/nz_0p25.cfdb', f'{S}/probe.cfdb')
sess = ds._blt._remote_session; ri = ds._blt._remote_index
print('remote keys: air_temperature_2m', sum(1 for k in ri.keys() if k.startswith('air_temperature_2m!')),
      '| air_temperature', sum(1 for k in ri.keys() if k.startswith('air_temperature!')))
keys = sorted(k for k in ri.keys() if k.startswith('air_temperature!'))
def get(key, a=None, b=None):
    r = sess.get_object(key, range_start=a, range_end=b); assert r.status in (200, 206); return len(r.data)
i = 0
def take(n):
    global i
    ks = keys[i:i + n]; i += n; assert len(ks) == n, 'ran out of keys'; return ks
print(f'{"test":34s} {"requests":>8s} {"MB":>7s} {"seconds":>8s} {"MB/s":>7s} {"req/s":>7s}')
for threads in (1, 10):
    for label, n, rng in [('16 KB ranges', 20, (0, 15_999)), ('256 KB ranges', 20, (0, 255_999)), ('full objects', 20, None)]:
        ks = take(n)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(threads) as ex:
            nb = sum(ex.map(lambda k: get(k, *rng) if rng else get(k), ks))
        t = time.perf_counter() - t0
        print(f'{threads:2d} thread(s), {label:20s} {n:8d} {nb/1e6:7.1f} {t:8.2f} {nb/t/1e6:7.1f} {n/t:7.1f}')
ds.close()
