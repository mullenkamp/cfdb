"""
cfdb Benchmarking Suite

Usage:
    python -m benchmarks.run_benchmarks [OPTIONS]

Options:
    --tier {small,medium,large,all}           Data size tier (default: small)
    --backends cfdb,netcdf4,zarr              Comma-separated (default: all)
    --scenarios write,read,iterate,filesize   Comma-separated (default: all)
    --output-json PATH                        Save results to JSON file
    --iterations N                            Override default iteration count
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from benchmarks.config import DATA_TIERS
from benchmarks.utils import generate_data, format_results_table, save_results_json
from benchmarks.scenarios.write import benchmark_write_full, benchmark_write_append
from benchmarks.scenarios.read import benchmark_read_full, benchmark_read_slice_aligned, benchmark_read_slice_unaligned
from benchmarks.scenarios.iterate import benchmark_iterate_chunks, benchmark_iterate_rechunked, benchmark_groupby_7day
from benchmarks.scenarios.filesize import measure_filesize


def get_available_backends():
    """Return backend classes for available dependencies."""
    backends = {}

    try:
        from benchmarks.backends.cfdb_backend import CfdbBackend
        backends['cfdb'] = CfdbBackend
    except ImportError as e:
        print(f"Skipping cfdb backend: {e}")

    try:
        from benchmarks.backends.netcdf4_backend import NetCDF4Backend
        backends['netcdf4'] = NetCDF4Backend
    except ImportError as e:
        print(f"Skipping netcdf4 backend: {e}")

    try:
        from benchmarks.backends.zarr_backend import ZarrBackend
        backends['zarr'] = ZarrBackend
    except ImportError as e:
        print(f"Skipping zarr backend: {e}")

    return backends


def run_scenarios(backend, file_path, scenarios, n_iterations):
    """Run selected scenarios for a backend, return results dict."""
    results = {}

    if 'write' in scenarios:
        print(f"  {backend.name}: write_full...", flush=True)
        results['write_full'] = benchmark_write_full(backend, n_iterations)
        print(f"  {backend.name}: write_append...", flush=True)
        results['write_append'] = benchmark_write_append(backend, n_iterations)

    if 'read' in scenarios or 'iterate' in scenarios or 'filesize' in scenarios:
        # Create a dataset for read/iterate/filesize benchmarks
        fp = backend.write_full()

        if 'read' in scenarios:
            print(f"  {backend.name}: read_full...", flush=True)
            results['read_full'] = benchmark_read_full(backend, fp, n_iterations)
            print(f"  {backend.name}: read_slice_aligned...", flush=True)
            results['read_slice_aligned'] = benchmark_read_slice_aligned(backend, fp, n_iterations)
            print(f"  {backend.name}: read_slice_unaligned...", flush=True)
            results['read_slice_unaligned'] = benchmark_read_slice_unaligned(backend, fp, n_iterations)

        if 'iterate' in scenarios:
            print(f"  {backend.name}: iterate_chunks...", flush=True)
            results['iterate_chunks'] = benchmark_iterate_chunks(backend, fp, n_iterations)
            print(f"  {backend.name}: iterate_rechunked...", flush=True)
            results['iterate_rechunked'] = benchmark_iterate_rechunked(backend, fp, n_iterations)
            print(f"  {backend.name}: groupby_7day...", flush=True)
            results['groupby_7day'] = benchmark_groupby_7day(backend, fp, n_iterations)

        if 'filesize' in scenarios:
            results['file_size'] = measure_filesize(backend, fp)

        backend.cleanup(fp)

    return results


def main():
    parser = argparse.ArgumentParser(description='cfdb Benchmarking Suite')
    parser.add_argument('--tier', choices=['small', 'medium', 'large', 'all'], default='small')
    parser.add_argument('--backends', default='all', help='Comma-separated: cfdb,netcdf4,zarr')
    parser.add_argument('--scenarios', default='all', help='Comma-separated: write,read,iterate,filesize')
    parser.add_argument('--output-json', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=None)
    args = parser.parse_args()

    available = get_available_backends()

    if args.backends == 'all':
        backend_names = list(available.keys())
    else:
        backend_names = [b.strip() for b in args.backends.split(',')]
        missing = [b for b in backend_names if b not in available]
        if missing:
            print(f"Warning: backends not available: {missing}")
        backend_names = [b for b in backend_names if b in available]

    if args.scenarios == 'all':
        scenarios = ['write', 'read', 'iterate', 'filesize']
    else:
        scenarios = [s.strip() for s in args.scenarios.split(',')]

    if args.tier == 'all':
        tiers = list(DATA_TIERS.keys())
    else:
        tiers = [args.tier]

    all_results = {}

    for tier_name in tiers:
        tier_config = DATA_TIERS[tier_name]
        n_iter = args.iterations or tier_config['n_iterations']

        shape_str = f"{tier_config['lat_size']}x{tier_config['lon_size']}x{tier_config['time_size']}"
        print(f"\n{'=' * 60}")
        print(f"Tier: {tier_name} ({shape_str}), {n_iter} iteration(s)")
        print(f"{'=' * 60}")

        print("Generating test data...", flush=True)
        data = generate_data(tier_config)

        tmp_base = Path(tempfile.mkdtemp(prefix='cfdb_bench_'))
        try:
            tier_results = {}

            for bname in backend_names:
                BackendClass = available[bname]
                backend = BackendClass()
                backend_dir = tmp_base / bname
                backend_dir.mkdir(parents=True, exist_ok=True)
                backend.setup(backend_dir, data)

                print(f"\nRunning {bname}...", flush=True)
                tier_results[bname] = run_scenarios(backend, None, scenarios, n_iter)

            all_results[tier_name] = tier_results
            format_results_table(tier_results, tier_name, tier_config)

        finally:
            shutil.rmtree(tmp_base, ignore_errors=True)

    if args.output_json:
        save_results_json(all_results, args.output_json)


if __name__ == '__main__':
    main()
