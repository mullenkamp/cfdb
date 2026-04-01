from benchmarks.utils import run_benchmark


def benchmark_iterate_chunks(backend, file_path, n_iterations):
    """Benchmark iterating over all storage chunks."""
    return run_benchmark(lambda: backend.iterate_chunks(file_path), n_iterations)


def benchmark_iterate_rechunked(backend, file_path, n_iterations):
    """Benchmark iterating with a rechunked shape (full spatial, 7 time steps)."""
    return run_benchmark(lambda: backend.iterate_rechunked(file_path), n_iterations)


def benchmark_groupby_7day(backend, file_path, n_iterations):
    """Benchmark grouping by 7-day periods with mean reduction."""
    return run_benchmark(lambda: backend.groupby_7day(file_path), n_iterations)
