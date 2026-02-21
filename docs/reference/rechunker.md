# Rechunker

The `Rechunker` class provides on-the-fly rechunking without modifying the stored data. Access it via `variable.rechunker()`.

::: cfdb.support_classes.Rechunker
    options:
      show_root_heading: true
      show_source: false
      members:
        - guess_chunk_shape
        - rechunk
        - calc_n_chunks
        - calc_n_reads_rechunker
        - calc_ideal_read_chunk_shape
        - calc_ideal_read_chunk_mem
        - calc_source_read_chunk_shape
