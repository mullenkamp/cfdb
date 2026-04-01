from benchmarks.utils import run_benchmark


def benchmark_read_full(backend, file_path, n_iterations):
    """Benchmark reading the entire data variable."""
    return run_benchmark(lambda: backend.read_full(file_path), n_iterations)


def benchmark_read_slice_aligned(backend, file_path, n_iterations):
    """Benchmark reading a chunk-aligned slice."""
    return run_benchmark(lambda: backend.read_slice_aligned(file_path), n_iterations)


def benchmark_read_slice_unaligned(backend, file_path, n_iterations):
    """Benchmark reading a non-aligned slice crossing chunk boundaries."""
    return run_benchmark(lambda: backend.read_slice_unaligned(file_path), n_iterations)
