"""Remote cost of alternative chunk shapes for a ts_ortho (station) dataset on an UNGROUPED remote
(one object, one GET, per chunk). Compressed chunk sizes are computed from the real data with the
dataset's own dtype encoding and compressor; GET counts per query are exact for the shape; time is
projected with the rates measured on the same host (GETs/s cold, MB/s). Reads the local copy in
station blocks — never the whole array."""
import sys, math, warnings, numpy as np
from cfdb import open_dataset
warnings.simplefilter('ignore')
path = sys.argv[1]
GETS_PER_S, MB_PER_S = float(sys.argv[2]), float(sys.argv[3])
ds = open_dataset(path); v = ds['streamflow']; dt = v.dtype; comp = v.compressor
P, T = v.shape
shapes = [(1, 25000), (1, 8760), (10, 8760), (141, 8760), (141, 2190), (141, 720), (141, 168)]
# compressed size of every non-empty chunk, per shape; one station block (all time) at a time
sizes = {s: {} for s in shapes}
for p0 in range(0, P, 10):                       # 10 stations x all hours = ~7 MB decoded
    blk = v[p0:p0 + 10, :].data
    for (sp, st) in shapes:
        for i in range(p0 - p0 % sp, min(p0 + 10, P), sp):
            for j in range(0, T, st):
                sub_p = slice(max(i, p0) - p0, min(i + sp, p0 + 10, P) - p0)
                piece = blk[sub_p, j:j + st]
                if np.isnan(piece).all():
                    continue
                full = np.full((sp, st), np.nan, 'float32')
                full[max(i, p0) - i: max(i, p0) - i + piece.shape[0], :piece.shape[1]] = piece
                key = (i // sp, j // st)
                sizes[(sp, st)].setdefault(key, np.full((sp, st), np.nan, 'float32'))
                cur = sizes[(sp, st)][key]
                cur[max(i, p0) - i: max(i, p0) - i + piece.shape[0], :piece.shape[1]] = piece
    del blk
nbytes = {s: {k: len(comp.compress(dt.dumps(a))) for k, a in d.items()} for s, d in sizes.items()}
del sizes
queries = {
    'latest 7 days, all stations':  (range(P), range(T - 168, T)),
    'latest 30 days, all stations': (range(P), range(T - 720, T)),
    'full history, one station':    (range(70, 71), range(T)),
    'full history, 10 stations':    (range(60, 70), range(T)),
    'whole dataset':                (range(P), range(T)),
}
print(f'projected with {GETS_PER_S} GETs/s and {MB_PER_S} MB/s (cold, 10 threads, measured on this host)')
print(f'{"chunk shape":14s} {"median KB":>9s} ' + ''.join(f'{q[:22]:>24s}' for q in queries) + f'{"objects rewritten per hourly update":>38s}')
for s in shapes:
    b = nbytes[s]; sp, st = s
    cells = ''
    for q, (pr, tr) in queries.items():
        keys = {(i // sp, j // st) for i in (pr.start, pr.stop - 1) for j in (tr.start, tr.stop - 1)}
        ks = [(a, c) for a in range(pr.start // sp, (pr.stop - 1) // sp + 1) for c in range(tr.start // st, (tr.stop - 1) // st + 1) if (a, c) in b]
        n = len(ks); mb = sum(b[k] for k in ks) / 1e6
        cells += f'{n:6d} GET {mb:6.1f}MB {n / GETS_PER_S + mb / MB_PER_S:5.1f}s'
    upd = len({(i // sp, (T - 1) // st) for i in range(P) if (i // sp, (T - 1) // st) in b})
    print(f'{str(s):14s} {np.median(list(b.values()))/1e3:9.0f} {cells}{upd:38d}')
ds.close()
