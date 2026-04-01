import time
import json
import statistics
import numpy as np
from pathlib import Path


class Timer:
    """Context manager for timing operations."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def run_benchmark(func, n_iterations, **kwargs):
    """Run a benchmark function n_iterations times, return timing stats."""
    times = []
    for _ in range(n_iterations):
        with Timer() as t:
            func(**kwargs)
        times.append(t.elapsed)

    return {
        'mean': statistics.mean(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0.0,
        'min': min(times),
        'max': max(times),
        'n': len(times),
    }


def generate_data(tier_config):
    """Generate test data arrays for a given tier configuration."""
    rng = np.random.default_rng(42)

    lat_data = np.linspace(-90, 90, tier_config['lat_size'], dtype='float32')
    lon_data = np.linspace(-180, 180, tier_config['lon_size'], dtype='float32')
    time_data = np.arange(
        np.datetime64('2020-01-01'),
        np.datetime64('2020-01-01') + tier_config['time_size'],
        dtype='datetime64[D]',
    )

    var_data = rng.uniform(
        0, 10000,
        size=(tier_config['lat_size'], tier_config['lon_size'], tier_config['time_size']),
    ).astype('float32')

    extra_time_data = np.arange(
        time_data[-1] + np.timedelta64(1, 'D'),
        time_data[-1] + np.timedelta64(1 + tier_config['append_time_size'], 'D'),
        dtype='datetime64[D]',
    )
    extra_var_data = rng.uniform(
        0, 10000,
        size=(tier_config['lat_size'], tier_config['lon_size'], tier_config['append_time_size']),
    ).astype('float32')

    chunk_shape = tier_config['chunk_shape']

    # Aligned slice: exactly one chunk
    aligned_sel = (
        slice(0, chunk_shape[0]),
        slice(0, chunk_shape[1]),
        slice(0, chunk_shape[2]),
    )

    # Unaligned slice: crosses chunk boundaries
    unaligned_sel = (
        slice(5, 5 + chunk_shape[0]),
        slice(10, 10 + chunk_shape[1]),
        slice(0, min(chunk_shape[2], tier_config['time_size'])),
    )

    # Rechunked shape: full spatial, 7 time steps
    rechunked_shape = {
        'latitude': tier_config['lat_size'],
        'longitude': tier_config['lon_size'],
        'time': 7,
    }

    return {
        'lat_data': lat_data,
        'lon_data': lon_data,
        'time_data': time_data,
        'var_data': var_data,
        'extra_time_data': extra_time_data,
        'extra_var_data': extra_var_data,
        'chunk_shape': chunk_shape,
        'aligned_sel': aligned_sel,
        'unaligned_sel': unaligned_sel,
        'rechunked_shape': rechunked_shape,
    }


def format_time(seconds):
    """Format seconds for display."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}us"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds:.3f}s"


def format_size(bytes_val):
    """Format bytes for display."""
    if bytes_val < 1024:
        return f"{bytes_val}B"
    elif bytes_val < 1024 ** 2:
        return f"{bytes_val / 1024:.1f}KB"
    elif bytes_val < 1024 ** 3:
        return f"{bytes_val / 1024 ** 2:.1f}MB"
    else:
        return f"{bytes_val / 1024 ** 3:.2f}GB"


def format_results_table(tier_results, tier_name, tier_config):
    """Print a formatted table comparing backends across operations."""
    backends = list(tier_results.keys())
    if not backends:
        return

    shape_str = f"{tier_config['lat_size']}x{tier_config['lon_size']}x{tier_config['time_size']}"
    print(f"\nTier: {tier_name} ({shape_str})")
    print("=" * (22 + 16 * len(backends)))

    header = f"{'Operation':<22}"
    for b in backends:
        header += f"{b:>16}"
    print(header)
    print("-" * (22 + 16 * len(backends)))

    # Collect all operation names across backends
    all_ops = []
    for b in backends:
        for op in tier_results[b]:
            if op not in all_ops:
                all_ops.append(op)

    for op in all_ops:
        row = f"{op:<22}"
        for b in backends:
            if op in tier_results[b]:
                result = tier_results[b][op]
                if op == 'file_size':
                    cell = format_size(result)
                    row += f"{cell:>16}"
                else:
                    mean_str = format_time(result['mean'])
                    if result['std'] > 0:
                        cell = f"{mean_str} +/-{format_time(result['std'])}"
                    else:
                        cell = mean_str
                    row += f"{cell:>16}"
            else:
                row += f"{'N/A':>16}"
        print(row)

    print()


def save_results_json(all_results, path):
    """Save full results to JSON file."""
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {path}")
