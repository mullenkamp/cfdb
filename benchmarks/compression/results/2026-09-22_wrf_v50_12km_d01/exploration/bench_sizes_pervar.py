"""Ratio vs block size, PER VARIABLE (no pooling), all 33 vars incl. the trivial ones, for
zstd-1 and numpy shuffle+zstd-1. Also the whole-file projected ratio at each block size."""
import json
import numpy as np
from bench import CODECS
from bench_numpy import C as CN
meta = json.load(open('chunks/meta.json'))
FIN = {'zstd-1': CODECS['zstd-1 (current)'], 'shuf+zstd-1': CN['numpy bitop-shuffle + zstd-1']}
SHAPES = [(24,1,324,277), (24,1,81,138), (24,1,20,34), (24,1,10,17), (24,1,5,8), (1,1,324,277), (1,1,40,34), (1,1,10,8)]
MAX_BYTES = 20e6

def blocks(a, shp):
    t, _, y, x = shp
    for i in range(0, a.shape[0], t):
        for j in range(0, a.shape[2], y):
            for k in range(0, a.shape[3], x):
                b = a[i:i+t, :, j:j+y, k:k+x]
                if b.shape == shp:
                    yield np.ascontiguousarray(b)

res = {}   # var -> shape -> codec -> (raw, comp)
for v in meta:
    stack = np.load(f'chunks/{v}.npy'); res[v] = {}
    for shp in SHAPES:
        res[v][shp] = {}
        for cname, (comp, _) in FIN.items():
            raw = cb = 0
            for a in stack:
                for b in blocks(a, shp):
                    if raw > MAX_BYTES: break
                    raw += b.nbytes; cb += len(comp(b))
            res[v][shp][cname] = (raw, cb)
    print('.', end='', flush=True)
print()
for cname in FIN:
    print(f'\n=== {cname}: per-variable ratio vs block shape (elements per block in header) ===')
    print(f'{"var":34s} {"dtype":8s}' + ''.join(f'{int(np.prod(s)):>9d}' for s in SHAPES))
    for v in meta:
        print(f'{v:34s} {meta[v]["enc_dtype"]:8s}' + ''.join(f'{res[v][s][cname][0]/res[v][s][cname][1]:9.1f}' for s in SHAPES))
    # whole-file projection: weight each var by its full raw size
    tot_raw = sum(sum(m['raw_bytes']) for m in meta.values())
    print(f'{"WHOLE FILE (33 vars) ratio":43s}' + ''.join(f'{tot_raw / sum(sum(meta[v]["raw_bytes"]) * res[v][s][cname][1] / res[v][s][cname][0] for v in meta):9.2f}' for s in SHAPES))
    print(f'{"WHOLE FILE projected MB":43s}' + ''.join(f'{sum(sum(meta[v]["raw_bytes"]) * res[v][s][cname][1] / res[v][s][cname][0] for v in meta)/1e6:9.1f}' for s in SHAPES))
