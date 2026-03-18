def measure_filesize(backend, file_path):
    """Measure on-disk file size in bytes."""
    return backend.get_file_size(file_path)
