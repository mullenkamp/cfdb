"""Can pure numpy (no new dependency) match c-blosc's shuffle / blosc2's shuffle+bytedelta?"""
import json, time, sys
import numpy as np, zstandard as zstd, blosc
from bench import CODECS
blosc.set_nthreads(1)

# --- bit-op byte-plane split (little-endian planes, low byte first) instead of a transpose
def planes_split(a):
    dt = a.dtype; n = dt.itemsize
    u = a.reshape(-1).view(f'u{n}') if n > 1 else a.reshape(-1).view('u1')
    if n == 1:
        return [u]
    return [(u >> (8 * i)).astype('u1') if i else u.astype('u1') for i in range(n)]  # astype truncates = & 0xFF

def planes_join(planes, dt, sh):
    dt = np.dtype(dt); n = dt.itemsize
    if n == 1:
        return planes[0].view(dt).reshape(sh)
    u = planes[-1].astype(f'u{n}')
    for i in range(n - 2, -1, -1):
        u <<= 8; u |= planes[i]
    return u.view(dt).reshape(sh)

def np_bitshuf_zstd(level, delta=False):
    c = zstd.ZstdCompressor(level=level); d = zstd.ZstdDecompressor()
    def comp(a):
        ps = planes_split(a)
        if delta:
            ps = [np.diff(p, prepend=np.uint8(0)) for p in ps]  # wrapping uint8 delta per plane
        return c.compress(b''.join(p.tobytes() for p in ps))
    def decomp(b, dt, sh):
        raw = d.decompress(b); n = np.dtype(dt).itemsize; m = len(raw) // n
        ps = [np.frombuffer(raw, 'u1', count=m, offset=i * m) for i in range(n)]
        if delta:
            ps = [np.cumsum(p, dtype='u1') for p in ps]
        return planes_join(ps, dt, sh)
    return comp, decomp

def blosc1(shuffle, level=1):
    return (lambda a: blosc.compress(a.tobytes(), typesize=a.itemsize, clevel=level, shuffle=shuffle, cname='zstd'),
            lambda b, dt, sh: np.frombuffer(blosc.decompress(b), dt).reshape(sh))

C = {
    'zstd-1 (current)': CODECS['zstd-1 (current)'],
    'numpy transpose-shuffle + zstd-1 (prev)': CODECS['npshuffle+zstd-1'],
    'numpy bitop-shuffle + zstd-1': np_bitshuf_zstd(1),
    'numpy bitop-shuffle + bytedelta + zstd-1': np_bitshuf_zstd(1, delta=True),
    'numpy bitop-shuffle + bytedelta + zstd-3': np_bitshuf_zstd(3, delta=True),
    'blosc1 zstd-1 shuffle (zero-dep pkg)': blosc1(blosc.SHUFFLE),
    'blosc2 zstd-1 shuffle': CODECS['blosc2 zstd-1 shuffle'],
    'blosc2 zstd-1 shuffle+bytedelta': CODECS['blosc2 zstd-1 shuffle+bytedelta'],
}
meta = json.load(open('chunks/meta.json'))
res = json.load(open('results_all.json')); base = res['zstd-1 (current)']
hard = [v for v in meta if base[v]['raw'] / base[v]['comp'] < 5]
if __name__ == '__main__':
  for cname, (comp, decomp) in C.items():
    raw = cb = ct = dt = 0; rawa = cba = 0
    for v in meta:
        for a in np.load(f'chunks/{v}.npy'):
            bc = bd = 1e9
            for _ in range(2):
                t0 = time.perf_counter(); b = comp(a); bc = min(bc, time.perf_counter() - t0)
                t0 = time.perf_counter(); o = decomp(b, str(a.dtype), a.shape); bd = min(bd, time.perf_counter() - t0)
            assert o.dtype == a.dtype and np.array_equal(o, a), (cname, v)
            rawa += a.nbytes; cba += len(b)
            if v in hard: raw += a.nbytes; cb += len(b); ct += bc; dt += bd
    print(f'{cname:44s} HARD ratio={raw/cb:5.2f} size={cb/1e6:6.1f} MB comp={raw/ct/1e6:5.0f} MB/s decomp={raw/dt/1e6:5.0f} MB/s | ALL file={cba/1e6:6.1f} MB', flush=True)
