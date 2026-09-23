"""Benchmark chunk codecs on a real cfdb dataset.

Usage:
    uv run python -m benchmarks.compression.codec_bench PATH.cfdb [OPTIONS]

Options:
    --codecs a,b,c          Codec names from codecs.CODECS (default: core trio; 'all' = every registered)
    --vars a,b,c            Data variables to include (default: all)
    --max-bytes-per-var N   Cap raw bytes read per variable (default: no cap)
    --reps N                Timing repetitions per chunk, min is kept (default: 3); one untimed
                            warm-up call per codec precedes the first timed chunk
    --hard-threshold R      Variables whose stored ratio is below R form the HARD subset (default: 5)
    --output-json PATH      Save per-variable results

Every chunk passes ``gate.RoundTripGate`` (bit-identical, input unmutated, C-contiguous,
no reused output buffer). String/geometry variables are skipped with a notice. Set OMP_NUM_THREADS=1 for a fair
single-thread comparison when optional OpenMP-built codecs are installed. Throughput is raw
(uncompressed) MB/s. Optional codecs: ``uv run --with blosc --with pcodec --with blosc2 ...``.
"""
import argparse
import json
import time

from benchmarks.compression.codecs import CODECS, DEFAULT
from benchmarks.compression.chunks import open_ro, iter_var_chunks, var_info, is_fixed_width
from benchmarks.compression.gate import RoundTripGate


def bench_dataset(path, codec_names, var_names=None, max_bytes=None, reps=3):
    ds = open_ro(path)
    try:
        var_names = var_names or list(ds.data_var_names)
        skipped = [v for v in var_names if not is_fixed_width(ds[v])]
        for v in skipped:
            print(f'{v:40s} skipped: {var_info(ds, v)["enc_dtype"]} is not a fixed-width dtype')
        var_names = [v for v in var_names if v not in skipped]
        info = {v: var_info(ds, v) for v in var_names}
        results = {c: {} for c in codec_names}
        gates = {c: RoundTripGate() for c in codec_names}
        warmed = set()
        stored = {}
        for v in var_names:
            recs = {c: dict(raw=0, comp=0, ct=0.0, dt=0.0, n=0) for c in codec_names}
            raw_total = stored_total = 0
            for arr, nbytes in iter_var_chunks(ds, v):
                if max_bytes is not None and raw_total >= max_bytes:
                    break
                raw_total += arr.nbytes
                stored_total += nbytes
                for c in codec_names:
                    comp, decomp = CODECS[c]
                    if c not in warmed:  # first call pays allocation/page-fault costs: keep it out of the timings
                        decomp(comp(arr), str(arr.dtype), arr.shape); warmed.add(c)
                    ref = gates[c].snapshot(arr)
                    bc = bd = float('inf')
                    for _ in range(reps):
                        t0 = time.perf_counter(); buf = comp(arr); bc = min(bc, time.perf_counter() - t0)
                    for _ in range(reps):
                        t0 = time.perf_counter(); out = decomp(buf, str(arr.dtype), arr.shape); bd = min(bd, time.perf_counter() - t0)
                    gates[c].check(arr, ref, out, f'{c}/{v}')
                    r = recs[c]
                    r['raw'] += arr.nbytes; r['comp'] += len(buf); r['ct'] += bc; r['dt'] += bd; r['n'] += 1
            stored[v] = dict(raw=raw_total, stored=stored_total)
            for c in codec_names:
                results[c][v] = recs[c]
            print(f'{v:40s} {raw_total/1e6:8.1f} MB raw, stored ratio {raw_total/max(stored_total,1):8.2f}', flush=True)
        return dict(path=str(path), compression=f'{ds.compression}-{ds.compression_level}', vars=info,
                    stored=stored, results=results)
    finally:
        ds.close()


def summarise(res, hard_threshold=5.0):
    stored, results = res['stored'], res['results']
    all_vars = list(stored)
    hard = [v for v in all_vars if stored[v]['raw'] / max(stored[v]['stored'], 1) < hard_threshold]
    raw_all = sum(stored[v]['raw'] for v in all_vars); st_all = sum(stored[v]['stored'] for v in all_vars)
    print(f'\nDataset: {res["path"]}  (stored as {res["compression"]})')
    print(f'{len(all_vars)} vars, raw {raw_all/1e6:.1f} MB, stored {st_all/1e6:.1f} MB (ratio {raw_all/max(st_all,1):.2f})')
    print(f'HARD = the {len(hard)} vars whose STORED ratio is < {hard_threshold} (a fixed set chosen from the file as stored, so every codec '
          f'is scored on the same population; ALL is the whole-file number): {", ".join(hard)}')

    def agg(c, vs):
        rr = results[c]
        return (sum(rr[v]['raw'] for v in vs), sum(rr[v]['comp'] for v in vs),
                sum(rr[v]['ct'] for v in vs), sum(rr[v]['dt'] for v in vs))

    hdr = f'{"codec":36s} {"ALL ratio":>9s} {"ALL MB":>8s} {"vs stored":>9s} | {"HARD ratio":>10s} {"comp MB/s":>10s} {"decomp MB/s":>12s}'
    print(hdr); print('-' * len(hdr))
    rows = []
    for c in results:
        rh, ch, cth, dth = agg(c, hard) if hard else (0, 1, 1, 1)
        ra, ca, _, _ = agg(c, all_vars)
        rows.append((c, rh / max(ch, 1), rh / max(cth, 1e-12) / 1e6, rh / max(dth, 1e-12) / 1e6, ra / max(ca, 1), ca / 1e6, ca / max(st_all, 1)))
    for c, ratio, cs, dsp, ra, mb, rel in sorted(rows, key=lambda x: -x[4]):
        print(f'{c:36s} {ra:9.2f} {mb:8.1f} {rel:9.2f} | {ratio:10.2f} {cs:10.0f} {dsp:12.0f}')

    print('\nPer-variable ratio:')
    cs = list(results)
    print(f'{"var":40s} {"dtype":8s} {"stored":>8s}' + ''.join(f'{c[:16]:>17s}' for c in cs))
    for v in all_vars:
        print(f'{v:40s} {res["vars"][v]["enc_dtype"]:8s} {stored[v]["raw"]/max(stored[v]["stored"],1):8.1f}'
              + ''.join(f'{results[c][v]["raw"]/max(results[c][v]["comp"],1):17.2f}' for c in cs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path')
    ap.add_argument('--codecs', default=None)
    ap.add_argument('--vars', default=None)
    ap.add_argument('--max-bytes-per-var', type=float, default=None)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--hard-threshold', type=float, default=5.0)
    ap.add_argument('--output-json', default=None)
    a = ap.parse_args()
    if a.codecs is None:
        codecs = DEFAULT
    elif a.codecs == 'all':
        codecs = list(CODECS)
    else:
        codecs = a.codecs.split(',')
    unknown = [c for c in codecs if c not in CODECS]
    if unknown:
        raise SystemExit(f'unknown codecs {unknown}; registered: {list(CODECS)}')
    res = bench_dataset(a.path, codecs, a.vars.split(',') if a.vars else None, a.max_bytes_per_var, a.reps)
    summarise(res, a.hard_threshold)
    if a.output_json:
        with open(a.output_json, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
