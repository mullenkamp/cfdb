"""Compression ratio (and speed) versus block size, per variable, on a real cfdb dataset.

Usage:
    uv run python -m benchmarks.compression.chunk_size_sweep PATH.cfdb [OPTIONS]

Options:
    --codecs a,b            Codec names (default: zstd-1,shuffle+zstd-1)
    --vars a,b              Data variables (default: all)
    --min-elements N        Stop shrinking blocks below this many elements (default: 64)
    --shrink-axis K         Shrink only chunk axis K (separates SHAPE effects from SIZE);
                            default halves the largest non-unit axis each step (aspect-preserving)
    --max-bytes-per-var N   Cap raw bytes read per variable per block shape (default: 20e6)
    --output-json PATH

Blocks are cut from the real stored chunks (numeric chunk order; the byte cap therefore keeps the
FIRST chunks), so the ladder starts at the variable's own chunk shape. Ratio is pooled over all
blocks of a shape for that variable. The whole-file projection is evaluated on a common grid of
block sizes (elements): each variable contributes the rung nearest that size, and a rung more
than 1.5x away is flagged — that only happens when variables have different chunk shapes.
Every block passes ``gate.RoundTripGate``; one untimed warm-up call precedes each (codec, shape).
"""
import argparse
import ast
import json
import time

import numpy as np

from benchmarks.compression.codecs import CODECS
from benchmarks.compression.chunks import open_ro, iter_var_chunks, var_info, is_fixed_width
from benchmarks.compression.gate import RoundTripGate


def block_ladder(chunk_shape, min_elements, shrink_axis=None):
    shapes = [tuple(chunk_shape)]
    while True:
        cur = list(shapes[-1])
        if shrink_axis is None:
            ax = int(np.argmax(cur))
        else:
            ax = shrink_axis
        if cur[ax] <= 1:
            break
        cur[ax] = max(1, cur[ax] // 2)
        if int(np.prod(cur)) < min_elements:
            break
        shapes.append(tuple(cur))
    return shapes


def iter_blocks(arr, shp):
    ranges = [range(0, s, b) for s, b in zip(arr.shape, shp)]
    idx = np.stack(np.meshgrid(*ranges, indexing='ij'), -1).reshape(-1, arr.ndim)
    for start in idx:
        sl = tuple(slice(int(s), int(s) + b) for s, b in zip(start, shp))
        b = arr[sl]
        if b.shape == shp:
            yield np.ascontiguousarray(b)


def sweep(path, codec_names, var_names=None, min_elements=64, shrink_axis=None, max_bytes=20e6):
    ds = open_ro(path)
    try:
        var_names = var_names or list(ds.data_var_names)
        var_names = [v for v in var_names if is_fixed_width(ds[v]) or print(f'{v:40s} skipped: not a fixed-width dtype')]
        out = {}
        gates = {c: RoundTripGate() for c in codec_names}
        for v in var_names:
            info = var_info(ds, v)
            ladder = block_ladder(info['chunk_shape'], min_elements, shrink_axis)
            per_shape = {}
            for shp in ladder:
                recs = {c: dict(raw=0, comp=0, ct=0.0, dt=0.0, n=0) for c in codec_names}
                done = False
                warmed = set()
                for arr, _ in iter_var_chunks(ds, v):
                    for b in iter_blocks(arr, shp):
                        if recs[codec_names[0]]['raw'] >= max_bytes:
                            done = True; break
                        for c in codec_names:
                            comp, decomp = CODECS[c]
                            if c not in warmed:
                                decomp(comp(b), str(b.dtype), b.shape); warmed.add(c)
                            ref = gates[c].snapshot(b)
                            t0 = time.perf_counter(); buf = comp(b); ct = time.perf_counter() - t0
                            t0 = time.perf_counter(); o = decomp(buf, str(b.dtype), b.shape); dt = time.perf_counter() - t0
                            gates[c].check(b, ref, o, f'{c}/{v}/{shp}')
                            r = recs[c]; r['raw'] += b.nbytes; r['comp'] += len(buf); r['ct'] += ct; r['dt'] += dt; r['n'] += 1
                    if done:
                        break
                per_shape[str(shp)] = recs
            out[v] = dict(info=info, full_raw_bytes=int(np.prod(info['shape'])) * np.dtype(info['enc_dtype']).itemsize, shapes=per_shape)
            print(f'{v:40s} {len(ladder)} block shapes: {ladder[0]} -> {ladder[-1]}', flush=True)
        return dict(path=str(path), codecs=codec_names, vars=out)
    finally:
        ds.close()


def summarise(res):
    codecs = res['codecs']
    vs = res['vars']
    # common grid: the union of every variable's rung sizes, largest first; each variable
    # contributes its nearest rung (in log space) and is flagged when that rung is > 1.5x away
    rungs = {v: {int(np.prod(ast.literal_eval(s))): s for s in r['shapes']} for v, r in vs.items()}
    grid = sorted({e for rr in rungs.values() for e in rr}, reverse=True)
    def nearest(v, e):
        es = np.array(list(rungs[v]))
        k = int(np.argmin(np.abs(np.log(es) - np.log(e))))
        return rungs[v][int(es[k])], max(es[k] / e, e / es[k])
    mism = {e: [v for v in vs if nearest(v, e)[1] > 1.5] for e in grid}
    total_raw = sum(r['full_raw_bytes'] for r in vs.values())
    for c in codecs:
        print(f'\n=== {c}: per-variable ratio vs block size (elements per block; "-" = no rung within 1.5x) ===')
        print(f'{"var":40s} {"dtype":8s}' + ''.join(f'{e:>10d}' for e in grid))
        for v, r in vs.items():
            cells = []
            for e in grid:
                key, off = nearest(v, e)
                rr = r['shapes'][key][c]
                cells.append(f'{"-":>10s}' if off > 1.5 else f'{rr["raw"]/max(rr["comp"],1):10.1f}')
            print(f'{v:40s} {r["info"]["enc_dtype"]:8s}' + ''.join(cells))
        proj = []
        for e in grid:
            tot = 0.0
            for v, r in vs.items():
                rr = r['shapes'][nearest(v, e)[0]][c]
                tot += r['full_raw_bytes'] * rr['comp'] / max(rr['raw'], 1)
            proj.append(total_raw / max(tot, 1))
        print(f'{"WHOLE FILE projected ratio":49s}' + ''.join(f'{p:10.2f}' for p in proj))
        print(f'{"WHOLE FILE projected MB":49s}' + ''.join(f'{total_raw/p/1e6:10.1f}' for p in proj))
    if any(mism.values()):
        print('\nNOTE: variables have different chunk shapes; at these grid sizes some contributed a rung > 1.5x away:')
        for e, bad in mism.items():
            if bad:
                print(f'  {e:>10d}: {", ".join(bad)}')
    print('\nThroughput MB/s (pooled over vars, nearest rung) by block size:')
    print(f'{"":49s}' + ''.join(f'{e:>10d}' for e in grid))
    for c in codecs:
        for key, lab in (('ct', 'compress'), ('dt', 'decompress')):
            cells = []
            for e in grid:
                raw = t = 0.0
                for v, r in vs.items():
                    rr = r['shapes'][nearest(v, e)[0]][c]; raw += rr['raw']; t += rr[key]
                cells.append(f'{raw/max(t,1e-12)/1e6:10.0f}')
            print(f'{c + " " + lab:49s}' + ''.join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path')
    ap.add_argument('--codecs', default='zstd-1,shuffle+zstd-1')
    ap.add_argument('--vars', default=None)
    ap.add_argument('--min-elements', type=int, default=64)
    ap.add_argument('--shrink-axis', type=int, default=None)
    ap.add_argument('--max-bytes-per-var', type=float, default=20e6)
    ap.add_argument('--output-json', default=None)
    a = ap.parse_args()
    codecs = a.codecs.split(',')
    unknown = [c for c in codecs if c not in CODECS]
    if unknown:
        raise SystemExit(f'unknown codecs {unknown}; registered: {list(CODECS)}')
    res = sweep(a.path, codecs, a.vars.split(',') if a.vars else None, a.min_elements, a.shrink_axis, a.max_bytes_per_var)
    summarise(res)
    if a.output_json:
        with open(a.output_json, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
