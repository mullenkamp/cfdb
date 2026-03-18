from benchmarks.utils import run_benchmark


def benchmark_write_full(backend, n_iterations):
    """Benchmark creating a dataset and writing all data."""
    paths = []

    def _run():
        p = backend.write_full()
        paths.append(p)

    result = run_benchmark(_run, n_iterations)

    # Cleanup all created files
    for p in paths:
        backend.cleanup(p)

    return result


def benchmark_write_append(backend, n_iterations):
    """Benchmark appending time steps to an existing dataset."""
    paths = []

    def _run():
        p = backend.write_full()
        paths.append(p)

    def _append():
        backend.write_append(paths[-1])

    results = []
    for _ in range(n_iterations):
        # Setup: create dataset (untimed)
        backend.write_full.__func__  # just checking it exists
        p = backend.write_full()
        paths.append(p)

        # Time only the append
        result = run_benchmark(lambda: backend.write_append(p), 1)
        results.append(result['mean'])
        backend.cleanup(p)

    import statistics
    return {
        'mean': statistics.mean(results),
        'std': statistics.stdev(results) if len(results) > 1 else 0.0,
        'min': min(results),
        'max': max(results),
        'n': len(results),
    }
