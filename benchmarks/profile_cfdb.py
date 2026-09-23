"""
cfdb Component Profiler

Breaks down cfdb operation timings into individual components:
  - Booklet key lookup
  - zstd decompression
  - dtype decode (from_bytes, astype, scale/offset)
  - rechunkit buffer management
  - full pipeline (iter_chunks, rechunked iter, groupby)

Two modes:

  Synthetic tiers (the original mode, unchanged): one random float32 variable at one chunk shape
  per tier; component totals for reads, rechunking and groupby.

      python -m benchmarks.profile_cfdb [--tier {small,medium,large}] [--max-mem BYTES] [--encoded]

  Real data, chunk-size ladder (``--source``): takes one variable of a real cfdb file, rewrites it
  into temporary files at a ladder of chunk shapes (from the stored chunk shape, or
  ``--start-chunk``, halving the largest axis each step down to ``--min-elements``), and reports
  per-chunk time for every step of the read and write paths, plus the full pipelines, at each size.

      python -m benchmarks.profile_cfdb --source PATH.cfdb --var NAME [--min-elements 1000]
                                        [--max-bytes 32e6] [--reps 3] [--output-json PATH]
                                        [--start-chunk 43,24576 --block-shape 43,49152]

  Read components:  keys (chunk-key generation), Booklet get, decompress, from_bytes, decode;
  full = ``iter_chunks()``.
  Write components, mirroring ``DataVariableView.set`` step for step: keys, Booklet get of the
  missing key, encode (ONCE per ``set()`` call, as ``set`` does it), assemble (blank-chunk copy +
  strided slice assignment), to_bytes, compress, Booklet set. Full pipelines: block writes (one
  ``set()`` per block, ``--block-shape``, default the stored chunk shape), the same data with one
  ``set()`` per target chunk, and block writes over existing chunks (read-modify-write). Keep every
  rung's extents dividing the block shape, or block writes become read-modify-writes.
  The close after a write (Booklet sync, then closing the file) is a per-FILE cost, reported
  separately; it grows with the file size and depends on filesystem state.

  ``sum`` is the sum of the components and ``resid`` = full - sum: work the components do not cover
  (the Python loop and input validation). A NEGATIVE residual beyond timing noise means the
  components are mis-measured. The check is coarse: review round cfdb-profile-1 put its detection
  floor at ~5-10 us/chunk, and it cannot see two errors that cancel. Rungs of >= 1 M elements are
  flagged: there the split is dominated by allocator/page-fault behaviour and is not reliable.

  All reads are WARM: the rewritten file sits in the OS page cache (dropping it needs root), so
  Booklet get measures an in-memory lookup + copy, not disk I/O. Each component is the min over
  ``--reps`` passes after an untimed warm-up pass, and drops each result before the next call so
  memory is reused the way the pipelines reuse it. Only the leading slice of the variable (along
  its first axis) that fits in ``--max-bytes`` of ENCODED data is used; component stages are held
  in memory for that slice. If the source is the local cache of a remote (EDataset) file, fetch
  every key first: missing chunks read back as fill values (cfdb warns with PartialDataWarning).
"""
import argparse
import gc
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import rechunkit as rk

from cfdb import open_dataset, dtypes, indexers
from cfdb.utils import make_var_chunk_key
from benchmarks.config import DATA_TIERS
from benchmarks.utils import generate_data, format_time, format_size
from benchmarks.compression.chunk_size_sweep import block_ladder


def profile_components(path, data, max_mem):
    """Profile individual cfdb components and print a breakdown."""
    lat_size = data['lat_data'].shape[0]
    lon_size = data['lon_data'].shape[0]
    time_size = data['time_data'].shape[0]
    chunk_shape = data['chunk_shape']
    rechunked_shape_tuple = (lat_size, lon_size, 7)

    results = {}

    with open_dataset(path) as ds:
        temp = ds['temperature']
        temp.load()

        is_encoded = temp.dtype.dtype_encoded is not None

        # Collect chunk slices for shape inference
        # iter_chunk_slices() was removed from cfdb (3c2b3b4); iter_chunks(include_data=False)
        # yields the same storage-chunk slices for a full variable
        all_chunk_slices = list(temp.iter_chunks(include_data=False))

        # --- 1. Booklet key lookup (no decompression, no decode) ---
        t0 = time.perf_counter()
        raw_chunks = []
        for cs in all_chunk_slices:
            key = _make_key(temp, cs)
            raw = temp._blt.get(key)
            raw_chunks.append(raw)
        t1 = time.perf_counter()
        n_chunks = len(raw_chunks)
        results['booklet_lookup'] = t1 - t0

        # --- 2. zstd decompression ---
        t0 = time.perf_counter()
        decompressed = []
        for raw in raw_chunks:
            decompressed.append(temp.compressor.decompress(raw))
        t1 = time.perf_counter()
        results['zstd_decompress'] = t1 - t0

        # --- 3. loads (deserialize bytes to decoded numpy array) ---
        # Decompressed bytes always contain a full storage chunk,
        # so use chunk_shape for deserialization (cfdb slices after)
        t0 = time.perf_counter()
        arrays = []
        for raw_bytes in decompressed:
            arrays.append(temp.dtype.loads(raw_bytes, chunk_shape))
        t1 = time.perf_counter()
        results['loads'] = t1 - t0

        if is_encoded:
            # --- 4a. dtype decode (astype + scale/offset) on small storage chunks ---
            # Re-deserialize as encoded arrays to isolate decode cost
            encoded_arrays = []
            for raw_bytes in decompressed:
                encoded_arrays.append(temp.dtype.from_bytes(raw_bytes, chunk_shape))

            t0 = time.perf_counter()
            for arr in encoded_arrays:
                temp.dtype.decode(arr)
            t1 = time.perf_counter()
            results['decode_small_chunks'] = t1 - t0

            # --- 4b. dtype decode on large output chunks (simulating rechunker output) ---
            big_arr = np.zeros(rechunked_shape_tuple, dtype=encoded_arrays[0].dtype)
            n_output = -(-time_size // 7)  # ceil division
            t0 = time.perf_counter()
            for _ in range(n_output):
                temp.dtype.decode(big_arr)
            t1 = time.perf_counter()
            results['decode_large_chunks'] = t1 - t0
        else:
            results['decode_small_chunks'] = 0.0
            results['decode_large_chunks'] = 0.0

        # --- 5. rechunkit buffer management (mock source, no I/O) ---

        # Use the storage dtype (encoded if applicable, else decoded)
        if is_encoded:
            mock_dtype = temp.dtype.dtype_encoded
        else:
            mock_dtype = temp.dtype.dtype_decoded
        sample_chunk = np.zeros(chunk_shape, dtype=mock_dtype)

        def mock_source(sel):
            return sample_chunk[:sel[0].stop - sel[0].start,
                                :sel[1].stop - sel[1].start,
                                :sel[2].stop - sel[2].start]

        t0 = time.perf_counter()
        for _ in rk.rechunker(mock_source, (lat_size, lon_size, time_size),
                              mock_dtype, chunk_shape, rechunked_shape_tuple,
                              max_mem):
            pass
        t1 = time.perf_counter()
        results['rechunkit_buffer_mgmt'] = t1 - t0

        # --- 6. Full pipeline: storage chunk iteration ---
        t0 = time.perf_counter()
        total = 0
        for _, chunk_data in temp.iter_chunks():
            total += chunk_data.size
        t1 = time.perf_counter()
        results['full_iter_storage'] = t1 - t0

        # --- 7. Full pipeline: rechunked iteration ---
        rechunked_dict = data['rechunked_shape']
        t0 = time.perf_counter()
        total = 0
        for _, chunk_data in temp.iter_chunks(rechunked_dict, max_mem=max_mem):
            total += chunk_data.size
        t1 = time.perf_counter()
        results['full_iter_rechunked'] = t1 - t0

        # --- 8. Full pipeline: groupby 7D ---
        t0 = time.perf_counter()
        count = 0
        for _, chunk_data in temp.groupby({'time': '7D'}, max_mem=max_mem):
            chunk_data.mean()
            count += 1
        t1 = time.perf_counter()
        results['full_groupby_7day'] = t1 - t0

        # --- Rechunker read stats ---
        rechunker = temp.rechunker()
        n_reads, n_writes = rechunker.calc_n_reads_rechunker(rechunked_shape_tuple, max_mem)
        ideal_shape = rechunker.calc_ideal_read_chunk_shape(rechunked_shape_tuple)
        source_shape = rechunker.calc_source_read_chunk_shape(rechunked_shape_tuple, max_mem)

    return results, {
        'n_chunks': n_chunks,
        'n_rechunker_reads': n_reads,
        'n_rechunker_writes': n_writes,
        'ideal_read_shape': ideal_shape,
        'source_read_shape': source_shape,
        'is_encoded': is_encoded,
        'total_elements': total,
    }


def _make_key(var, chunk_slices):
    """Reconstruct the Booklet key for a chunk."""
    coord_origins = var.get_coord_origins()
    starts = tuple(
        chunk_slices[i].start + coord_origins[i]
        for i in range(len(chunk_slices))
    )
    return make_var_chunk_key(var.name, starts)


def print_results(results, stats, tier_name, tier_config, max_mem):
    shape_str = f"{tier_config['lat_size']}x{tier_config['lon_size']}x{tier_config['time_size']}"
    chunk_str = f"{tier_config['chunk_shape']}"

    print(f"\ncfdb Component Profile — {tier_name} ({shape_str}), chunks {chunk_str}")
    print(f"max_mem={format_size(max_mem)}, encoded={stats['is_encoded']}")
    print("=" * 60)

    print(f"\n  Storage chunks:       {stats['n_chunks']}")
    print(f"  Rechunker reads:      {stats['n_rechunker_reads']}")
    print(f"  Rechunker writes:     {stats['n_rechunker_writes']}")
    print(f"  Ideal read shape:     {stats['ideal_read_shape']}")
    print(f"  Actual read shape:    {stats['source_read_shape']}")

    print(f"\n{'Component':<30} {'Time':>10}")
    print("-" * 42)

    components = [
        ('Booklet lookup', 'booklet_lookup'),
        ('zstd decompress', 'zstd_decompress'),
        ('loads (deserialize+decode)', 'loads'),
        ('decode (small chunks)', 'decode_small_chunks'),
        ('decode (large chunks)', 'decode_large_chunks'),
        ('rechunkit buffer mgmt', 'rechunkit_buffer_mgmt'),
    ]
    for label, key in components:
        print(f"  {label:<28} {format_time(results[key]):>10}")

    print()
    print(f"{'Full Pipeline':<30} {'Time':>10}")
    print("-" * 42)

    pipelines = [
        ('iter_chunks (storage)', 'full_iter_storage'),
        ('iter_chunks (rechunked)', 'full_iter_rechunked'),
        ('groupby 7D', 'full_groupby_7day'),
    ]
    for label, key in pipelines:
        print(f"  {label:<28} {format_time(results[key]):>10}")

    print()


# =============================================================================
# Real-data chunk-size ladder
# =============================================================================

READ_COMPONENTS = ('keys', 'get', 'decompress', 'from_bytes', 'decode')
WRITE_COMPONENTS = ('keys', 'get_missing', 'encode', 'assemble', 'to_bytes', 'compress', 'set')
SPLIT_UNRELIABLE = 1_000_000  # elements/chunk from which the component split is allocator-dominated


def _drain(it):
    """Consume and DROP each result, as the real pipelines do: memory is freed and reused chunk
    by chunk. Holding every output in a list instead makes each call page-fault fresh memory,
    which the pipelines never pay."""
    for _ in it:
        pass


def _min_pass(func, reps):
    """Untimed warm-up, then the min over ``reps`` timed passes (gc collected before each)."""
    func()
    best = float('inf')
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - t0)
    return best


def _leading_slice(var, max_bytes):
    """Length along axis 0 whose ENCODED bytes fit in max_bytes (at least one row)."""
    dt = var.dtype
    itemsize = np.dtype(dt.dtype_encoded if dt.dtype_encoded is not None else dt.dtype_decoded).itemsize
    row = math.prod(var.shape[1:]) * itemsize
    return max(1, min(var.shape[0], int(max_bytes // row)))


def _create_like(src_ds, src_var, path, chunk_shape, n0):
    """Empty dataset holding one variable defined like src_var (same dtype object, compression,
    coords restricted to the leading n0 along axis 0), stored at chunk_shape."""
    view = src_ds.select({src_var.coord_names[0]: slice(0, n0)})
    ds = open_dataset(path, flag='n', dataset_type=src_ds.dataset_type,
                      compression=src_ds.compression, compression_level=src_ds.compression_level)
    for c in src_var.coord_names:
        ds.create.coord.like(c, view[c], copy_data=True)
    dv = ds.create.data_var.generic(src_var.name, src_var.coord_names, dtype=src_var.dtype, chunk_shape=chunk_shape)
    return ds, dv


def _chunk_keys(ds, name):
    """Every stored chunk key of one variable, in numeric chunk-start order."""
    prefix = f'{name}!'
    return sorted((k for k in ds._blt.keys() if k.startswith(prefix)),
                  key=lambda k: tuple(int(x) for x in k.split('!', 1)[1].split('.')))


def _timed_close(ds):
    """(Booklet sync, rest of close) in seconds."""
    t0 = time.perf_counter()
    ds._blt.sync()
    t1 = time.perf_counter()
    ds.close()
    return t1 - t0, time.perf_counter() - t1


def profile_rung(src_ds, src_var, blocks, n0, chunk_shape, tmp, reps):
    """All components and full pipelines for one chunk shape.

    ``blocks`` is the source data as (global slices, decoded array) pairs at the SOURCE chunk
    shape. Two write patterns are timed: block writes (one ``set`` per source block, as an ingest
    writing large arrays does) and one ``set`` per target chunk (the loop the cfdb skill
    recommends: "your write loop should mirror your chunk_shape"). Components mirror the BLOCK
    write, step for step, including where each step is paid: encode once per ``set`` call, the
    rest once per target chunk.
    """
    name = src_var.name
    path = tmp / 'rung.cfdb'
    res = {}
    dt0 = src_var.dtype
    encoded = dt0.dtype_encoded is not None

    # per-chunk pieces for the one-set-per-chunk pattern: (global slices, contiguous data)
    pieces = []
    for sl, data in blocks:
        for tc, _, _ in indexers.slices_to_chunks_keys(sl, name, chunk_shape):
            g = tuple(slice(s.start + t.start, s.start + t.stop) for s, t in zip(sl, tc))
            pieces.append((g, np.ascontiguousarray(data[tc])))

    def write(pattern):
        """Fresh file, then the timed set loop; the close is timed apart (per FILE), with Booklet
        sync and the file close separated. The close grows with the file size and depends on
        filesystem state (~3 ms for a 6 MB file, 25-50 ms for 30-64 MB ones on ext4 here)."""
        if path.exists():
            path.unlink()
        ds, dv = _create_like(src_ds, src_var, path, chunk_shape, n0)
        t0 = time.perf_counter()
        for sl, data in (blocks if pattern == 'block' else pieces):
            dv[sl] = data
        t = time.perf_counter() - t0
        return (t,) + _timed_close(ds)

    for key, pattern in (('write_per_chunk', 'chunk'), ('write_full', 'block')):  # block last: its file stays
        write(pattern)  # warm-up
        runs = [write(pattern) for _ in range(reps)]
        res[key] = min(r[0] for r in runs)
    res['sync_fresh'] = min(r[1] for r in runs)  # per FILE, block pattern
    res['close_fresh'] = min(r[2] for r in runs)

    def overwrite():
        ds = open_dataset(path, flag='w')
        dv = ds[name]
        t0 = time.perf_counter()
        for sl, data in blocks:
            dv[sl] = data
        t = time.perf_counter() - t0
        _timed_close(ds)
        return t
    res['write_full_rmw'] = min(overwrite() for _ in range(reps))

    # ---- read components + full read on the written file (warm page cache)
    ds = open_dataset(path, flag='r')
    dv = ds[name]
    dt, comp, blt = dv.dtype, dv.compressor, ds._blt
    full_sel = tuple(slice(0, s) for s in dv.shape)
    keys = _chunk_keys(ds, name)
    n = len(keys)
    raws = [blt.get(k) for k in keys]
    decs = [comp.decompress(r) for r in raws]
    if encoded:
        encs = [dt.from_bytes(b, chunk_shape) for b in decs]
    else:
        encs = [np.frombuffer(bytearray(b), dtype=dt.dtype_decoded).reshape(chunk_shape) for b in decs]

    res['keys_read'] = _min_pass(lambda: _drain(indexers.slices_to_chunks_keys(full_sel, name, chunk_shape)), reps)
    res['get'] = _min_pass(lambda: _drain(blt.get(k) for k in keys), reps)
    res['decompress'] = _min_pass(lambda: _drain(comp.decompress(r) for r in raws), reps)
    if encoded:
        res['from_bytes'] = _min_pass(lambda: _drain(dt.from_bytes(b, chunk_shape) for b in decs), reps)
        res['decode'] = _min_pass(lambda: _drain(dt.decode(e) for e in encs), reps)
    else:  # dtype.loads for an unpacked dtype is exactly this frombuffer(bytearray(...))
        res['from_bytes'] = _min_pass(lambda: _drain(np.frombuffer(bytearray(b), dtype=dt.dtype_decoded).reshape(chunk_shape) for b in decs), reps)
        res['decode'] = 0.0

    def full_read():
        for _ in dv.iter_chunks():
            pass
    res['read_full'] = _min_pass(full_read, reps)
    blank = dv._make_blank_chunk_array(False) if encoded else dv._make_blank_chunk_array()
    ds.close()

    # ---- write components, mirroring DataVariableView.set for the BLOCK write
    block_pairs = [list(indexers.slices_to_chunks_keys(sl, name, chunk_shape)) for sl, _ in blocks]
    res['keys_write'] = _min_pass(lambda: _drain(k for sl, _ in blocks for k in indexers.slices_to_chunks_keys(sl, name, chunk_shape)), reps)
    if encoded:  # set() encodes the whole input ONCE, before its chunk loop
        res['encode'] = _min_pass(lambda: _drain(dt.encode(d) for _, d in blocks), reps)
        enc_blocks = [dt.encode(d) for _, d in blocks]
    else:
        res['encode'] = 0.0
        enc_blocks = [d for _, d in blocks]

    def assemble_all():  # blank copy + strided slice assignment from the block, per target chunk
        for eb, pairs in zip(enc_blocks, block_pairs):
            for tc, sc, _ in pairs:
                b = blank.copy()
                b[sc] = eb[tc]
                yield b
    res['assemble'] = _min_pass(lambda: _drain(assemble_all()), reps)
    chunks = list(assemble_all())
    if encoded:
        res['to_bytes'] = _min_pass(lambda: _drain(dt.to_bytes(c) for c in chunks), reps)
        payloads = [dt.to_bytes(c) for c in chunks]
    else:
        res['to_bytes'] = _min_pass(lambda: _drain(dt.dumps(c) for c in chunks), reps)
        payloads = [dt.dumps(c) for c in chunks]
    res['compress'] = _min_pass(lambda: _drain(comp.compress(b) for b in payloads), reps)
    wkeys = [k for pairs in block_pairs for _, _, k in pairs]
    wraws = [comp.compress(b) for b in payloads]

    # Booklet get of a missing key, then set, into a fresh file opened the way cfdb opens it
    # (write mode). Calls only: sync happens at close and is reported per file.
    scratch = tmp / 'scratch.cfdb'
    def booklet_pass():
        if scratch.exists():
            scratch.unlink()
        sds, _ = _create_like(src_ds, src_var, scratch, chunk_shape, n0)
        sblt = sds._blt
        t0 = time.perf_counter()
        for k in wkeys:
            sblt.get(k)
        t1 = time.perf_counter()
        for k, r in zip(wkeys, wraws):
            sblt.set(k, r)
        t2 = time.perf_counter()
        sds.close()
        return t1 - t0, t2 - t1
    booklet_pass()
    runs = [booklet_pass() for _ in range(reps)]
    res['get_missing'] = min(r[0] for r in runs)
    res['set'] = min(r[1] for r in runs)

    res['keys'] = res['keys_read']  # the READ table's key component; the WRITE table uses keys_write
    res['read_sum'] = sum(res[c] for c in READ_COMPONENTS)
    res['write_sum'] = sum(res['keys_write' if c == 'keys' else c] for c in WRITE_COMPONENTS)
    res['n_chunks'] = n
    res['n_write_chunks'] = len(wkeys)
    res['elements'] = math.prod(chunk_shape)
    res['chunk_bytes'] = res['elements'] * encs[0].dtype.itemsize
    res['stored_bytes'] = sum(len(r) for r in raws)
    path.unlink()
    scratch.unlink(missing_ok=True)
    return res


def profile_ladder(src_path, var_name, min_elements=1000, max_bytes=32e6, reps=3, start_chunk=None, block_shape=None):
    """``start_chunk``: first rung of the ladder (default: the stored chunk shape). Needed to reach
    shapes wider than the stored chunk, e.g. multi-station chunks when stations are stored one per
    chunk. ``block_shape``: shape of the blocks written per ``set()`` call (default: the stored chunk
    shape). Blocks narrower than the target chunks turn block writes into read-modify-writes, so set
    it at least as wide as ``start_chunk`` when widening."""
    src_ds = open_dataset(src_path)
    tmp = Path(tempfile.mkdtemp(prefix='cfdb_profile_ladder_'))
    try:
        src_var = src_ds[var_name]
        n0 = _leading_slice(src_var, max_bytes)
        view = src_ds.select({src_var.coord_names[0]: slice(0, n0)})[var_name]
        # the source data, decoded, bounded by max_bytes (encoded); copied because rechunked
        # iteration may yield views into a reused buffer
        if block_shape is None:
            blocks = [(sl, np.array(data)) for sl, data in view.iter_chunks()]
        else:
            spec = {c: min(b, s) for c, b, s in zip(src_var.coord_names, block_shape, view.shape)}
            blocks = [(sl, np.array(data)) for sl, data in view.iter_chunks(chunk_shape=spec)]
        first = start_chunk if start_chunk is not None else src_var.chunk_shape
        start = tuple(min(c, s) for c, s in zip(first, view.shape))
        ladder = block_ladder(start, min_elements)
        out = dict(source=str(src_path), var=var_name, shape=list(view.shape), source_chunk=list(src_var.chunk_shape),
                   start_chunk=list(start), block_shape=list(block_shape) if block_shape else list(src_var.chunk_shape),
                   dtype=str(src_var.dtype.dtype_decoded), dtype_encoded=str(src_var.dtype.dtype_encoded),
                   compression=f'{src_ds.compression}-{src_ds.compression_level}', reps=reps, rungs={})
        for shp in ladder:
            out['rungs'][str(shp)] = profile_rung(src_ds, src_var, blocks, n0, shp, tmp, reps)
            r = out['rungs'][str(shp)]
            print(f'  {str(shp):22s} {r["n_chunks"]:6d} chunks  read {r["read_full"]/r["n_chunks"]*1e6:9.1f} us/chunk  '
                  f'write {r["write_full"]/r["n_chunks"]*1e6:9.1f} us/chunk', flush=True)
        return out
    finally:
        src_ds.close()
        shutil.rmtree(tmp, ignore_errors=True)


def print_ladder(out):
    print(f"\ncfdb per-chunk profile — {out['var']} {tuple(out['shape'])} "
          f"{out['dtype']} (stored as {out['dtype_encoded']}), {out['compression']}, warm page cache, min of {out['reps']}")
    print(f"source: {out['source']}")
    rungs = list(out['rungs'].values())
    real = math.prod(out['shape'])

    def table(title, comps, full_key, sum_key, n_key, extra=()):
        cols = list(comps) + ['sum', 'full', 'resid', 'resid%'] + [e[0] for e in extra]
        print(f"\n{title} — microseconds per chunk (! = component split unreliable at >= {SPLIT_UNRELIABLE} elements)")
        print(f"{'elements':>9s} {'chunks':>7s} " + ''.join(f'{c:>12s}' for c in cols))
        for r in rungs:
            n = r[n_key]
            vals = [r[('keys_write' if (c == 'keys' and 'WRITE' in title) else c)] / n * 1e6 for c in comps]
            sm = r[sum_key] / n * 1e6
            full = r[full_key] / n * 1e6
            cells = ''.join(f'{v:12.2f}' for v in vals + [sm, full, full - sm])
            cells += f'{(full - sm) / full * 100:11.0f}%' + ''.join(f'{r[e[1]] / n * 1e6:12.2f}' for e in extra)
            flag = '!' if r['elements'] >= SPLIT_UNRELIABLE else ' '
            print(f"{r['elements']:8d}{flag} {n:7d} " + cells)

    table('READ (iter_chunks)', READ_COMPONENTS, 'read_full', 'read_sum', 'n_chunks')
    table('WRITE (one set() per source block)', WRITE_COMPONENTS, 'write_full', 'write_sum', 'n_write_chunks',
          extra=(('per_chunk', 'write_per_chunk'), ('rmw', 'write_full_rmw')))
    print("  per_chunk = the same data written with one set() per target chunk; rmw = block write over existing chunks")

    print("\nFull pipelines, ns per element — per REAL element (the variable's) / per STORED element (incl. edge padding)")
    print(f"{'elements':>9s} {'stored/real':>11s} {'read':>15s} {'write (block)':>17s} {'write (per chunk)':>19s} {'write (rmw)':>15s}")
    for r in rungs:
        stored = r['n_chunks'] * r['elements']
        cells = ''
        for k in ('read_full', 'write_full', 'write_per_chunk', 'write_full_rmw'):
            cells += f"{r[k] / real * 1e9:9.2f} /{r[k] / stored * 1e9:6.2f}"
        print(f"{r['elements']:9d} {stored / real:11.3f} {cells}")

    print("\nPer FILE after a fresh block write: " + ', '.join(
        f"{r['elements']}: Booklet sync {r['sync_fresh']*1e3:.1f} ms + close {r['close_fresh']*1e3:.1f} ms" for r in rungs))
    print("Residual (full - sum) is a coarse check: mutation tests in review round cfdb-profile-1 put its detection floor at\n"
          "~5-10 us/chunk, and it cannot see errors that cancel. A positive residual is uncovered work (the Python loop).")


def main():
    parser = argparse.ArgumentParser(description='cfdb Component Profiler')
    parser.add_argument('--tier', choices=['small', 'medium', 'large'], default='small')
    parser.add_argument('--max-mem', type=int, default=2**31,
                        help='Rechunker max memory in bytes (default: 2GB)')
    parser.add_argument('--encoded', action='store_true',
                        help='Use encoded dtype (uint32 with precision=1)')
    parser.add_argument('--source', default=None, help='real cfdb file: switches to the chunk-size ladder mode')
    parser.add_argument('--var', default=None, help='data variable in --source')
    parser.add_argument('--min-elements', type=int, default=1000, help='smallest chunk on the ladder (default 1000)')
    parser.add_argument('--max-bytes', type=float, default=32e6, help='encoded bytes of the variable to use (default 32e6)')
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--start-chunk', default=None, help='first ladder rung, comma-separated (default: stored chunk shape)')
    parser.add_argument('--block-shape', default=None, help='shape written per set() call, comma-separated (default: stored chunk shape)')
    parser.add_argument('--output-json', default=None)
    args = parser.parse_args()

    if args.source is not None:
        if args.var is None:
            parser.error('--source needs --var')
        parse = lambda x: tuple(int(v) for v in x.split(',')) if x else None
        out = profile_ladder(args.source, args.var, args.min_elements, args.max_bytes, args.reps,
                             parse(args.start_chunk), parse(args.block_shape))
        print_ladder(out)
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(out, f, indent=1)
        return

    tier_config = DATA_TIERS[args.tier]
    shape_str = f"{tier_config['lat_size']}x{tier_config['lon_size']}x{tier_config['time_size']}"
    print(f"Generating {args.tier} data ({shape_str})...", flush=True)
    data = generate_data(tier_config)

    if args.encoded:
        data_dtype = dtypes.dtype(data['var_data'].dtype, 1, 0, 10000)
    else:
        data_dtype = dtypes.dtype('float32')

    tmp = Path(tempfile.mkdtemp(prefix='cfdb_profile_'))
    path = tmp / 'profile.cfdb'

    try:
        print("Creating dataset...", flush=True)
        with open_dataset(path, flag='n') as ds:
            ds.create.coord.lat(data=data['lat_data'],
                                chunk_shape=(data['chunk_shape'][0],))
            ds.create.coord.lon(data=data['lon_data'],
                                chunk_shape=(data['chunk_shape'][1],))
            ds.create.coord.time(data=data['time_data'],
                                 dtype=data['time_data'].dtype)
            dv = ds.create.data_var.generic(
                'temperature',
                ('latitude', 'longitude', 'time'),
                data_dtype,
                chunk_shape=data['chunk_shape'],
            )
            dv[:] = data['var_data']

        print(f"File size: {format_size(path.stat().st_size)}")
        print("Profiling...", flush=True)

        results, stats = profile_components(path, data, args.max_mem)
        print_results(results, stats, args.tier, tier_config, args.max_mem)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
