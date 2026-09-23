"""Pull every stored chunk out of the cfdb file as the ENCODED array cfdb feeds to zstd
(i.e. exactly the bytes the compressor sees), plus the on-disk compressed size."""
import json
import numpy as np
import cfdb

src = '/home/mike/data/wrf/sst/v50_12km_wvt_8_regions/cfdb_cache/d01.cfdb'
ds = cfdb.open_dataset(src)
meta = {}
for name in ds.data_var_names:
    v = ds[name]
    dt = v.dtype
    enc_dtype = dt.dtype_encoded if dt.dtype_encoded is not None else dt.dtype_decoded
    keys = sorted(k for k in ds._blt.keys() if k.startswith(f'{name}!'))
    arrs, csizes = [], []
    for k in keys:
        b = ds._blt.get(k)
        csizes.append(len(b))
        raw = v.compressor.decompress(b)
        arrs.append(np.frombuffer(raw, dtype=enc_dtype).reshape(v.chunk_shape))
    stack = np.stack(arrs)
    np.save(f'chunks/{name}.npy', stack)
    meta[name] = dict(enc_dtype=str(np.dtype(enc_dtype)), dec_dtype=str(dt.dtype_decoded),
                      precision=dt.precision, offset=dt.offset, fillvalue=getattr(dt, 'fillvalue', None),
                      chunk_shape=list(v.chunk_shape), n_chunks=len(keys), raw_bytes=[a.nbytes for a in arrs],
                      zstd1_bytes=csizes)
    print(f'{name:35s} {len(keys)} chunks raw={stack.nbytes/1e6:8.1f} MB  stored={sum(csizes)/1e6:7.1f} MB  ratio={stack.nbytes/sum(csizes):5.2f}')
ds.close()
json.dump(meta, open('chunks/meta.json', 'w'), indent=1)
tot_raw = sum(sum(m['raw_bytes']) for m in meta.values()); tot_c = sum(sum(m['zstd1_bytes']) for m in meta.values())
print(f'TOTAL raw={tot_raw/1e6:.1f} MB stored={tot_c/1e6:.1f} MB ratio={tot_raw/tot_c:.2f}')
