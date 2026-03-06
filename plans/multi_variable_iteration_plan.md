# Plan: Multi-Variable `iter_chunks()` and `map()` on DatasetBase

## Context

Users currently iterate chunks of a single variable at a time. For multi-variable calculations (e.g., combining temperature and humidity), they must manually coordinate iteration. Adding dataset-level `iter_chunks()` and `map()` lets users iterate once and get aligned data for all variables.

The updated rechunkit now guarantees **canonical C-order yield ordering** regardless of source chunk shape (via a reordering buffer with `_canon_idx` + `pending` dict). This means per-variable rechunkers with the same `target_chunk_shape` and `target_shape` can be **zipped directly** — their target slices are guaranteed to match.

### Design decisions (from user)

- `chunk_shape` as `{coord_name: int}` dict
- Methods on `DatasetBase` (shared by Dataset and DatasetView)
- `map()` parallelism: main process reads data, workers compute

### Why zipping rechunkers now works

The old rechunkit yielded target chunks in an order determined by source-target chunk alignment (LCM-based read groups). Variables with different storage chunk shapes produced different yield orderings, making zip unreliable.

The updated rechunkit adds a canonical ordering layer:

```python
# rechunkit/main.py lines 449-502
n_chunks_per_dim = tuple(ceil(s / c) for s, c in zip(target_shape, target_chunk_shape))

def _canon_idx(chunk_slices):
    idx = 0
    for s, c, nc in zip(chunk_slices, target_chunk_shape, n_chunks_per_dim):
        idx = idx * nc + (s.start // c)
    return idx

pending = {}
next_idx = 0
# ... yields in canonical C-order, buffering out-of-order chunks
```

The yield order now depends **only** on `target_chunk_shape` and `target_shape` — independent of `source_chunk_shape` and `max_mem`. Two rechunker instances with the same target parameters produce identical target slices in identical order.

## Files to Modify

1. **`cfdb/support_classes.py`** — Update `Rechunker` for new rechunkit API, add `_ds_map_worker`
2. **`cfdb/tools.py`** — Update rechunkit API calls
3. **`cfdb/main.py`** — Add `import rechunkit`, add `iter_chunks()` and `map()` to `DatasetBase`
4. **`cfdb/tests/test_dataset.py`** — Add tests

## Step 1: Update rechunkit API calls (breaking signature changes)

The updated rechunkit moved `itemsize` from positional param #4 to keyword argument in `rechunker()`, and `guess_chunk_shape()` now takes `itemsize: int` instead of `dtype`.

### `cfdb/support_classes.py`

**`Rechunker.guess_chunk_shape()` (~line 78)** — pass itemsize not dtype:

```python
# OLD:
chunk_shape = rechunkit.guess_chunk_shape(self._var.shape, self._var.dtype_encoded, target_chunk_size)

# NEW:
itemsize = self._guess_itemsize()
chunk_shape = rechunkit.guess_chunk_shape(self._var.shape, itemsize, target_chunk_size)
```

**`Rechunker.rechunk()` (~lines 168, 175)** — move itemsize to keyword:

```python
# OLD (line 168, dtype_decoded path):
rechunkit1 = rechunkit.rechunker(func, self._var.shape, self._var.dtype.dtype_decoded, itemsize, self._var.chunk_shape, target_chunk_shape, max_mem, self._var._sel)

# NEW:
rechunkit1 = rechunkit.rechunker(func, self._var.shape, self._var.dtype.dtype_decoded, self._var.chunk_shape, target_chunk_shape, max_mem, self._var._sel, itemsize=itemsize)

# OLD (line 175, dtype_encoded path):
rechunkit1 = rechunkit.rechunker(func, self._var.shape, self._var.dtype.dtype_encoded, itemsize, self._var.chunk_shape, target_chunk_shape, max_mem, self._var._sel)

# NEW:
rechunkit1 = rechunkit.rechunker(func, self._var.shape, self._var.dtype.dtype_encoded, self._var.chunk_shape, target_chunk_shape, max_mem, self._var._sel, itemsize=itemsize)
```

### `cfdb/tools.py`

**Lines 212, 300** — `guess_chunk_shape`:

```python
# OLD:
chunk_shape = rechunkit.guess_chunk_shape(shape, dtype_encoded)

# NEW:
chunk_shape = rechunkit.guess_chunk_shape(shape, dtype_encoded.itemsize)
```

**Line 309** — `rechunker`:

```python
# OLD:
chunks_iter = rechunkit.rechunker(h5_reader.get, shape, dtype_encoded, dtype_encoded.itemsize, chunk_shape, chunk_shape, max_mem, var_sel)

# NEW:
chunks_iter = rechunkit.rechunker(h5_reader.get, shape, dtype_encoded, chunk_shape, chunk_shape, max_mem, var_sel, itemsize=dtype_encoded.itemsize)
```

### Calls that remain unchanged

These rechunkit calls have the same signature in the updated version — no changes needed:

- `rechunkit.chunk_range(starts, stops, chunk_shape)` — `indexers.py:277,291`, `support_classes.py:530,847`
- `rechunkit.calc_ideal_read_chunk_shape(source, target)` — `support_classes.py:86,94`
- `rechunkit.calc_ideal_read_chunk_mem(shape, itemsize)` — `support_classes.py:96`
- `rechunkit.calc_source_read_chunk_shape(source, target, itemsize, max_mem)` — `support_classes.py:115`
- `rechunkit.calc_n_chunks(shape, chunk_shape)` — `support_classes.py:121`
- `rechunkit.calc_n_reads_rechunker(shape, itemsize, source, target, max_mem, sel)` — `support_classes.py:142`
- `rechunkit.guess_chunk_shape(shape, itemsize, max)` — `utils.py:582,644` (already pass int itemsize)

## Step 2: Add `_ds_map_worker` to `support_classes.py`

Add below `_ChunkMapWrapper` class (~line 951). Top-level function required for multiprocessing pickling:

```python
def _ds_map_worker(args):
    func, target_chunk, var_data = args
    result = func(target_chunk, var_data)
    if result is not None:
        return (target_chunk, result)
    return None
```

## Step 3: Add `iter_chunks()` to `DatasetBase` in `main.py`

Add `import rechunkit` to imports. Add method after `select_loc` (~line 216):

```python
def iter_chunks(self, chunk_shape, include_data=False, data_vars=None, max_mem=2**27):
    """
    Iterate over aligned chunks of multiple data variables.

    Parameters
    ----------
    chunk_shape : dict
        {coord_name: int} — iteration chunk size per coordinate.
        Coords not listed use their full length (single step).
    include_data : bool
        If True, yield (target_chunk, var_data). If False, yield target_chunk only.
    data_vars : list of str, optional
        Which data variables to include. Default: all.
    max_mem : int
        Max memory in bytes for rechunker buffer per variable. Default 128 MB.

    Yields
    ------
    target_chunk : dict
        {coord_name: slice} — chunk positions in the iteration space.
    var_data : dict (if include_data)
        {var_name: ndarray} — decoded data for each variable at this position.
    """
    # Validate data vars
    if data_vars is None:
        data_vars = list(self.data_var_names)
    else:
        for name in data_vars:
            if name not in self.data_var_names:
                raise KeyError(f'{name} is not a data variable.')

    if not data_vars:
        return

    # Determine common coord_names — require all data vars share the same coords
    first_dv = self[data_vars[0]]
    common_coord_names = first_dv.coord_names
    for dv_name in data_vars[1:]:
        if self[dv_name].coord_names != common_coord_names:
            raise ValueError(
                f'All data variables must share the same coordinates. '
                f'{data_vars[0]} has {common_coord_names}, '
                f'{dv_name} has {self[dv_name].coord_names}.'
            )

    # Build target_chunk_shape: coords not in chunk_shape use full length
    target_chunk_shape = tuple(
        chunk_shape.get(cn, self[cn].shape[0])
        for cn in common_coord_names
    )

    if include_data:
        # Create per-variable rechunker generators
        rechunker_gens = []
        for dv_name in data_vars:
            dv = self[dv_name]
            r = dv.rechunker()
            rechunker_gens.append(r.rechunk(target_chunk_shape, max_mem))

        # Zip — canonical C-order guaranteed by rechunkit
        for items in zip(*rechunker_gens):
            write_chunk = items[0][0]
            target_chunk = {
                cn: sl for cn, sl in zip(common_coord_names, write_chunk)
            }
            var_data = {
                dv_name: item[1]
                for dv_name, item in zip(data_vars, items)
            }
            yield target_chunk, var_data
    else:
        # Just iterate positions using chunk_range
        starts = tuple(0 for _ in common_coord_names)
        stops = tuple(self[cn].shape[0] for cn in common_coord_names)
        for position in rechunkit.chunk_range(starts, stops, target_chunk_shape):
            yield {cn: sl for cn, sl in zip(common_coord_names, position)}
```

### How it works

1. **Validates** all data vars share the same `coord_names` (same names, same order).
2. **Builds** `target_chunk_shape` from the `chunk_shape` dict — coords not listed default to their full length.
3. **Creates** one `Rechunker.rechunk()` generator per variable, all with the same `target_chunk_shape`.
4. **Zips** the generators. Rechunkit's canonical ordering guarantees all generators yield the same target slices in the same order, regardless of per-variable `source_chunk_shape`.
5. **Yields** `(target_chunk_dict, var_data_dict)` at each position.

### Why this works for DatasetView

When `DatasetView.get(var_name)` returns a variable, it applies the view's selection via `self._dataset.get(var_name)[self._sel[var_name]]`, creating a `DataVariableView` with `_sel` set. The `Rechunker.rechunk()` passes `self._var._sel` to `rechunkit.rechunker()`, which adjusts `target_shape` accordingly. All variables in the same view sharing the same coords have the same `_sel` → same `target_shape` → same canonical ordering → zip works.

## Step 4: Add `map()` to `DatasetBase` in `main.py`

Add after `iter_chunks()`:

```python
def map(self, func, chunk_shape, data_vars=None, max_mem=2**27, n_workers=None):
    """
    Apply func to aligned chunks of multiple variables in parallel.

    Parameters
    ----------
    func : callable
        func(target_chunk, var_data) -> result or None.
        target_chunk: dict of {coord_name: slice}.
        var_data: dict of {var_name: ndarray}.
        Must be a top-level function (picklable).
    chunk_shape : dict
        {coord_name: int} for iteration chunk sizes.
    data_vars : list of str, optional
        Data variable names to include. Default: all.
    max_mem : int
        Max memory for rechunker buffer per variable. Default 128 MB.
    n_workers : int, optional
        Number of worker processes. Defaults to os.cpu_count().

    Yields
    ------
    tuple
        (target_chunk, result) pairs, as workers complete (unordered).
    """
    import multiprocessing
    import os

    if n_workers is None:
        n_workers = os.cpu_count()

    def work_iter():
        for target_chunk, var_data in self.iter_chunks(
            chunk_shape, include_data=True, data_vars=data_vars, max_mem=max_mem
        ):
            yield (func, target_chunk, var_data)

    with multiprocessing.Pool(n_workers) as pool:
        for result in pool.imap_unordered(sc._ds_map_worker, work_iter()):
            if result is not None:
                yield result
```

### How it works

1. **Main process** iterates positions via `iter_chunks()`, reading and assembling all variable data per position using rechunkers.
2. **Sends** `(func, target_chunk, var_data)` tuples to worker pool via `imap_unordered`.
3. **Workers** call `func(target_chunk, var_data)` and return results (or `None` to skip).
4. **Main process** yields `(target_chunk, result)` as workers complete. Order is non-deterministic (fastest worker first).

Memory per worker: one target chunk per variable (numpy arrays, already decoded). `func` is a top-level function reference — cheap to pickle.

## Step 5: Add Tests

Top-level picklable functions + tests in `cfdb/tests/test_dataset.py`:

```python
def _sum_two_vars(target_chunk, var_data):
    return var_data['var1'] + var_data['var2']

def _ds_return_none(target_chunk, var_data):
    return None


def test_dataset_iter_chunks():
    """Test dataset-level iter_chunks with multiple variables."""
    fp = tmp_path / 'ds_iter.cfdb'

    # Create dataset with 2 vars sharing lat/lon
    with cfdb.open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=np.linspace(-90, 90, 181, dtype='float32'))
        ds.create.coord.lon(data=np.linspace(-180, 180, 361, dtype='float32'))
        v1 = ds.create.data_var.generic('var1', ('latitude', 'longitude'), dtype='float32')
        v2 = ds.create.data_var.generic('var2', ('latitude', 'longitude'), dtype='float32')
        data1 = np.ones((181, 361), dtype='float32')
        data2 = np.ones((181, 361), dtype='float32') * 2
        v1[:] = data1
        v2[:] = data2

    with cfdb.open_dataset(fp) as ds:
        # Test with include_data=True — verify shapes and values
        total_elements = 0
        for target_chunk, var_data in ds.iter_chunks(
            {'latitude': 50, 'longitude': 100}, include_data=True
        ):
            lat_sl = target_chunk['latitude']
            lon_sl = target_chunk['longitude']
            expected1 = data1[lat_sl, lon_sl]
            expected2 = data2[lat_sl, lon_sl]
            np.testing.assert_array_equal(var_data['var1'], expected1)
            np.testing.assert_array_equal(var_data['var2'], expected2)
            total_elements += var_data['var1'].size
        assert total_elements == 181 * 361

        # Test with include_data=False — yields dicts of slices
        chunks = list(ds.iter_chunks({'latitude': 50, 'longitude': 100}))
        assert len(chunks) > 0
        assert isinstance(chunks[0], dict)
        assert 'latitude' in chunks[0]
        assert 'longitude' in chunks[0]

        # Test data_vars filter
        for _, var_data in ds.iter_chunks(
            {'latitude': 181}, include_data=True, data_vars=['var1']
        ):
            assert 'var1' in var_data
            assert 'var2' not in var_data

        # Test with DatasetView
        view = ds.select({'latitude': slice(10, 50)})
        for target_chunk, var_data in view.iter_chunks(
            {'latitude': 10}, include_data=True
        ):
            lat_sl = target_chunk['latitude']
            assert lat_sl.stop - lat_sl.start <= 10
            assert var_data['var1'].shape[0] <= 10

        # Test ValueError when coords differ (need a 3rd var with different coords)
        # Could create a dataset with time coord + a var with (lat, lon, time)
        # and attempt iter_chunks with both — should raise ValueError


def test_dataset_map():
    """Test dataset-level map with multiple variables."""
    fp = tmp_path / 'ds_map.cfdb'

    with cfdb.open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=np.linspace(-90, 90, 181, dtype='float32'))
        ds.create.coord.lon(data=np.linspace(-180, 180, 361, dtype='float32'))
        v1 = ds.create.data_var.generic('var1', ('latitude', 'longitude'), dtype='float32')
        v2 = ds.create.data_var.generic('var2', ('latitude', 'longitude'), dtype='float32')
        v1[:] = np.ones((181, 361), dtype='float32') * 3
        v2[:] = np.ones((181, 361), dtype='float32') * 7

    with cfdb.open_dataset(fp) as ds:
        # Sum two vars via map
        results = list(ds.map(
            _sum_two_vars, {'latitude': 50, 'longitude': 100}, n_workers=2
        ))
        assert len(results) > 0
        for target_chunk, result in results:
            np.testing.assert_array_equal(result, np.full_like(result, 10.0))

        # Test None return (skip)
        results = list(ds.map(
            _ds_return_none, {'latitude': 181}, n_workers=2
        ))
        assert len(results) == 0
```

## Usage Examples

```python
# Multi-variable iteration
with cfdb.open_dataset('weather.cfdb') as ds:
    for chunk, data in ds.iter_chunks({'time': 30}, include_data=True):
        temp = data['temperature']
        humidity = data['humidity']
        heat_index = compute_heat_index(temp, humidity)

# Parallel multi-variable processing
def compute_index(chunk, data):
    return data['temperature'] * 0.7 + data['humidity'] * 0.3

for chunk, result in ds.map(compute_index, {'latitude': 50, 'longitude': 100}, n_workers=4):
    output_var[tuple(chunk[cn] for cn in output_var.coord_names)] = result

# With DatasetView (filtered)
view = ds.select({'time': slice(0, 365)})
for chunk, data in view.iter_chunks({'time': 30}, include_data=True):
    process(data)

# Selective variables
for chunk, data in ds.iter_chunks({'time': 30}, include_data=True, data_vars=['temperature']):
    pass  # Only temperature loaded
```

## Current limitation

All data variables passed to `iter_chunks()` / `map()` must share the same coordinate set. Variables with different coordinate subsets (e.g., `temperature(lat, lon, time)` and `elevation(lat, lon)`) cannot be iterated together. This covers the most common use case. Mixed-coordinate support can be added in a future enhancement.

## Verification

```bash
# Update rechunkit dependency first (install from local)
uv pip install -e /home/mike/git/rechunkit

# Run existing tests to verify API updates don't break anything
uv run pytest cfdb/tests/test_dataset.py -v

# Run new tests
uv run pytest cfdb/tests/test_dataset.py -k "test_dataset_iter_chunks or test_dataset_map" -v
```
