"""cfdb core-path performance benchmark suite.

Measures read/write/rechunk/groupby throughput plus a many-small-operations
microbench on seeded ~50 MB datasets, so behavior-preserving changes can be
gated against performance regressions (and perf-round improvements measured).

Usage:
    uv run python benchmarks/bench_core.py --output benchmarks/results/baseline-YYYY-MM-DD.json
    uv run python benchmarks/bench_core.py --compare benchmarks/results/baseline.json benchmarks/results/after.json

Protocol: median of N_RUNS (default 5) wall-clock runs per case, fixed seeds,
datasets built once in a temp dir per invocation. Only APIs valid on both
pre- and post-fix trees are used.
"""
import argparse
import gc
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

import cfdb
from cfdb import dtypes

N_RUNS = 5
NY, NT = 1000, 6400                     # 51.2 MB float64
CS = (100, 640)


def _median_time(func):
    times = []
    for _ in range(N_RUNS):
        gc.collect()
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def build_datasets(base: Path):
    rng = np.random.default_rng(42)
    data = rng.normal(scale=10.0, size=(NY, NT))
    t0 = np.datetime64('2020-01-01T00', 'h')

    def create(path):
        with cfdb.open_dataset(path, flag='n') as ds:
            ds.create.coord.generic(name='y', data=np.arange(0.0, NY), chunk_shape=(CS[0],), step=1.0)
            ds.create.coord.time(data=t0 + np.arange(NT), dtype='datetime64[h]', chunk_shape=(CS[1],), step=True)
            v = ds.create.data_var.generic(name='v', coords=('y', 'time'),
                                           dtype=dtypes.dtype('float64'), chunk_shape=CS)
            v[:] = data
            vf = ds.create.data_var.generic(name='vf', coords=('y', 'time'),
                                            dtype=dtypes.dtype('float32', precision=2, min_value=-100, max_value=100),
                                            chunk_shape=CS)
            vf[:] = data.astype('float32')
            vi = ds.create.data_var.generic(name='vi', coords=('y', 'time'),
                                            dtype=dtypes.dtype('int32', min_value=0, max_value=60000),
                                            chunk_shape=CS)
            vi[:] = (np.abs(data) * 100).astype('int32')

    create(base / 'main.cfdb')
    create(base / 'prepended.cfdb')
    with cfdb.open_dataset(base / 'prepended.cfdb', flag='w') as ds:
        ds['y'].prepend(np.array([-1.0]))
        ds['time'].prepend(np.array([t0 - 1]))
    return data


def run_benchmarks(base: Path):
    results = {}
    mb = NY * NT * 8 / 2**20
    data = build_datasets(base)
    rng = np.random.default_rng(1)
    blocks = [(int(a), int(b)) for a, b in
              zip(rng.integers(0, NY - 37, size=200), rng.integers(0, NT - 41, size=200))]

    with cfdb.open_dataset(base / 'main.cfdb', flag='w') as ds:
        dv = ds['v']

        t = _median_time(lambda: dv.data)
        results['read_full_plain'] = {'median_s': t, 'mb_per_s': mb / t}

        def write_full():
            dv[:] = data
        t = _median_time(write_full)
        results['write_full_plain'] = {'median_s': t, 'mb_per_s': mb / t}

        for name in ('v', 'vf', 'vi'):
            var = ds[name]
            if name == 'v':
                block = data[:37, :41]
            elif name == 'vf':
                block = data[:37, :41].astype('float32')
            else:
                block = (np.abs(data[:37, :41]) * 100).astype('int32')

            def write_partial(var=var, block=block):
                for a, b in blocks:
                    var[a:a + 37, b:b + 41] = block
            t = _median_time(write_partial)
            results[f'write_partial_{name}'] = {'median_s': t, 'blocks_per_s': len(blocks) / t}

        def rechunk_aligned():
            for _wc, _d in dv.rechunker().rechunk((250, 320)):
                pass
        t = _median_time(rechunk_aligned)
        results['rechunk_aligned'] = {'median_s': t, 'mb_per_s': mb / t}

        def iter_storage():
            for _tc, _d in dv.iter_chunks():
                pass
        t = _median_time(iter_storage)
        results['iter_chunks_storage'] = {'median_s': t, 'mb_per_s': mb / t}

        def groupby_daily():
            for _sl, _d in dv.groupby({'time': 'D'}):
                pass
        t = _median_time(groupby_daily)
        results['groupby_daily_fast'] = {'median_s': t, 'mb_per_s': mb / t}

        # Non-divisor groupby amplification (P1's cfdb-level gate): daily
        # groups (chunk 24, non-divisor of storage 640) under a small max_mem
        # so the target's LCM band exceeds the budget.  Planned reads vs
        # stored chunks is the deterministic amplification metric; wall time
        # rides along.
        gb_mm = 2**22
        stored_chunks = (NY // CS[0]) * (NT // CS[1])
        n_reads, _n_writes = dv.rechunker().calc_n_reads_rechunker((NY, 24), gb_mm)
        results['groupby_amplification'] = {
            'planned_reads': n_reads,
            'stored_chunks': stored_chunks,
            'amplification': n_reads / stored_chunks,
            'max_mem_mb': gb_mm / 2**20,
        }

        def groupby_daily_tight():
            for _sl, _d in dv.groupby({'time': 'D'}, max_mem=gb_mm):
                pass
        t = _median_time(groupby_daily_tight)
        results['groupby_amplification']['median_s'] = t
        results['groupby_amplification']['mb_per_s'] = mb / t

        # Many-small-operations microbench: per-call selection overhead.
        small = data[:5, :5]

        def many_small_ops():
            for a, b in blocks[:100]:
                _ = dv[a:a + 5, b:b + 5].data
                _ = dv.loc[float(a):float(a + 4), :].shape
                dv[a:a + 5, b:b + 5] = small
        t = _median_time(many_small_ops)
        results['many_small_ops_100'] = {'median_s': t, 'ops_per_s': 300 / t}

    with cfdb.open_dataset(base / 'prepended.cfdb') as ds:
        dvp = ds['v']

        def rechunk_prepended():
            for _wc, _d in dvp.rechunker().rechunk((250, 320)):
                pass
        t = _median_time(rechunk_prepended)
        results['rechunk_prepended'] = {'median_s': t, 'mb_per_s': mb / t}

    return results


def compare(base_path, new_path):
    with open(base_path) as f:
        base = json.load(f)
    with open(new_path) as f:
        new = json.load(f)
    print(f"{'case':<26} {'metric':<12} {'base':>12} {'new':>12} {'delta':>8}")
    for case in base:
        if case not in new:
            continue
        for metric in base[case]:
            b, n = base[case][metric], new[case].get(metric)
            if n is None or not isinstance(b, (int, float)) or b == 0:
                continue
            delta = (n - b) / b * 100
            flag = '  <<<' if (metric == 'median_s' and delta > 10) else ''
            print(f"{case:<26} {metric:<12} {b:>12.4g} {n:>12.4g} {delta:>+7.1f}%{flag}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', help='write results JSON to this path')
    p.add_argument('--compare', nargs=2, metavar=('BASE', 'NEW'), help='compare two results files')
    args = p.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    tmp = Path(tempfile.mkdtemp(prefix='cfdb_bench_'))
    try:
        results = run_benchmarks(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"cfdb {cfdb.__version__}")
    for case, metrics in results.items():
        line = '  '.join(f"{k}={v:.4g}" for k, v in metrics.items())
        print(f"{case:<26} {line}")
    if args.output:
        import os
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=1)
        print(f"written: {args.output}")


if __name__ == '__main__':
    main()
