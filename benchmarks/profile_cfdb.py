"""
cfdb Component Profiler

Breaks down cfdb operation timings into individual components:
  - Booklet key lookup
  - zstd decompression
  - dtype decode (from_bytes, astype, scale/offset)
  - rechunkit buffer management
  - full pipeline (iter_chunks, rechunked iter, groupby)

Usage:
    python -m benchmarks.profile_cfdb [OPTIONS]

Options:
    --tier {small,medium,large}   Data size tier (default: small)
    --max-mem BYTES               Rechunker max memory in bytes (default: 2**29)
"""
import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from cfdb import open_dataset, dtypes
from benchmarks.config import DATA_TIERS
from benchmarks.utils import generate_data, format_time, format_size


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
        all_chunk_slices = list(temp.iter_chunk_slices())

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
        import rechunkit as rk

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
    from cfdb.utils import make_var_chunk_key
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


def main():
    parser = argparse.ArgumentParser(description='cfdb Component Profiler')
    parser.add_argument('--tier', choices=['small', 'medium', 'large'], default='small')
    parser.add_argument('--max-mem', type=int, default=2**31,
                        help='Rechunker max memory in bytes (default: 2GB)')
    parser.add_argument('--encoded', action='store_true',
                        help='Use encoded dtype (uint32 with precision=1)')
    args = parser.parse_args()

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
