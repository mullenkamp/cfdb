# DatasetRechunker

Assistant class for rechunking multiple variables in a dataset simultaneously. Created via [`ds.rechunker()`](dataset.md#rechunker).

All variables in a batch must share the same coordinates, shapes, and storage chunking to ensure synchronized iteration.

## Methods

### calc_ideal_read_chunk_mem(chunk_shape)

Calculate the total ideal memory (in bytes) required to process the entire batch without redundant reads.

Parameters:
- `chunk_shape` (dict): `{coord_name: int}` for target chunk sizes.

### calc_n_reads_rechunker(chunk_shape, max_mem=2\*\*29)

Calculate the number of read operations required for the batch.

Parameters:
- `chunk_shape` (dict): `{coord_name: int}` for target chunk sizes.
- `max_mem` (int): Total memory budget in bytes.

### rechunk(chunk_shape, max_mem=2\*\*29)

Synchronized multivariable rechunking generator.

Parameters:
- `chunk_shape` (dict): `{coord_name: int}` for target chunk sizes.
- `max_mem` (int): Total max memory budget in bytes for all variables combined.

Yields:
- `target_chunk` (dict): `{coord_name: slice}` — chunk positions.
- `var_data` (dict): `{var_name: ndarray}` — synchronized data blocks.
