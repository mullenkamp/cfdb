"""Side question: if cfdb kept the values as float32 (after its decimal rounding) instead of
packing to uint16, how would pcodec / blosc2 do on the float32 directly? Compressed bytes are
compared per ELEMENT so the uint16 and float32 pipelines are directly comparable."""
import json, time
import numpy as np
from bench import CODECS
meta = json.load(open('chunks/meta.json'))
VARS = ['air_temperature', 'surface_pressure', 'wind_speed', 'mixing_ratio', 'pwat']
codecs_u = ['zstd-1 (current)', 'blosc2 zstd-1 shuffle+bytedelta', 'pcodec L8 (default)']
codecs_f = ['zstd-1 (current)', 'blosc2 zstd-1 shuffle+bytedelta', 'pcodec L8 (default)', 'zfp lossless', 'fpzip lossless (f32 only)']
print(f'{"var":18s} {"pipeline":48s} {"bytes/elem":>10s} {"comp MB/s(raw f32)":>18s} {"decomp MB/s":>12s}')
for v in VARS:
    m = meta[v]; a = np.load(f'chunks/{v}.npy')[:2]
    # decode exactly as cfdb's DTypeTranscoder.decode does (float32 arithmetic)
    f = a.astype(np.float32); f[a == m['fillvalue']] = np.nan
    f = (f / np.float32(10 ** m['precision'])) + np.float32(m['offset'])
    assert np.isnan(f).sum() == (a == 0).sum()
    for c in codecs_u:
        comp, decomp = CODECS[c]; n = cb = ct = dt = 0
        for blk in a:
            t0 = time.perf_counter(); b = comp(blk); ct += time.perf_counter() - t0
            t0 = time.perf_counter(); o = decomp(b, str(blk.dtype), blk.shape); dt += time.perf_counter() - t0
            cb += len(b); n += blk.size
        print(f'{v:18s} {"uint16 packed + " + c:48s} {cb/n:10.3f} {n*4/ct/1e6:18.0f} {n*4/dt/1e6:12.0f}')
    for c in codecs_f:
        comp, decomp = CODECS[c]; n = cb = ct = dt = 0
        for blk in f:
            t0 = time.perf_counter(); b = comp(blk); ct += time.perf_counter() - t0
            t0 = time.perf_counter(); o = decomp(b, 'float32', blk.shape); dt += time.perf_counter() - t0
            assert np.array_equal(o, blk, equal_nan=True)
            cb += len(b); n += blk.size
        print(f'{v:18s} {"float32 direct + " + c:48s} {cb/n:10.3f} {n*4/ct/1e6:18.0f} {n*4/dt/1e6:12.0f}')
    print()
