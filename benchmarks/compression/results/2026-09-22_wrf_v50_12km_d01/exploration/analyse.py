import json
r1 = json.load(open('results_part1.json')); r2 = json.load(open('results_partial.json'))
res = {**r1, **r2}
json.dump(res, open('results_all.json', 'w'), indent=1)
meta = json.load(open('chunks/meta.json'))
base = res['zstd-1 (current)']
hard = [v for v in meta if base[v]['raw'] / base[v]['comp'] < 5]
trivial = [v for v in meta if v not in hard]
print('hard vars:', len(hard), ' trivial vars:', len(trivial), trivial)
raw_hard = sum(base[v]['raw'] for v in hard); base_hard = sum(base[v]['comp'] for v in hard)
print(f'hard subset raw={raw_hard/1e6:.0f} MB, zstd-1 stored={base_hard/1e6:.0f} MB of total stored {sum(base[v]["comp"] for v in meta)/1e6:.0f} MB\n')

def agg(cname, vs):
    rr = res[cname]
    vs = [v for v in vs if v in rr]
    raw = sum(rr[v]['raw'] for v in vs); comp = sum(rr[v]['comp'] for v in vs)
    ct = sum(rr[v]['ct'] for v in vs); dt = sum(rr[v]['dt'] for v in vs)
    return raw, comp, ct, dt

hdr = f'{"codec":42s} {"ratio":>6s} {"size MB":>8s} {"vs zstd1":>8s} {"comp MB/s":>10s} {"decomp MB/s":>12s} | {"ALL ratio":>9s} {"file MB":>8s}'
print('=== HARD subset (24 vars, non-trivial) | ALL 33 vars ==='); print(hdr); print('-'*len(hdr))
rows = []
for c in res:
    raw, comp, ct, dt = agg(c, hard)
    raw_a, comp_a, _, _ = agg(c, list(meta))
    rows.append((c, raw/comp, comp/1e6, comp/base_hard, raw/ct/1e6, raw/dt/1e6, raw_a/comp_a, comp_a/1e6))
for row in sorted(rows, key=lambda x: -x[1]):
    c, ratio, mb, rel, cs, ds, ra, fa = row
    flag = '' if not c.startswith('fpzip') else '  (ivt only)'
    print(f'{c:42s} {ratio:6.2f} {mb:8.1f} {rel:8.2f} {cs:10.0f} {ds:12.0f} | {ra:9.2f} {fa:8.1f}{flag}')

# per-variable ratio table for a shortlist
short = ['zstd-1 (current)', 'zstd-3', 'npshuffle+zstd-1', 'blosc2 zstd-1 shuffle', 'blosc2 zstd-1 shuffle+bytedelta', 'blosc2 lz4 shuffle', 'pcodec L4', 'pcodec L8 (default)', 'zfp lossless']
print('\n=== per-variable compression ratio (hard vars) ===')
print(f'{"var":34s} {"dtype":8s}' + ''.join(f'{s[:14]:>15s}' for s in short))
for v in hard:
    print(f'{v:34s} {meta[v]["enc_dtype"]:8s}' + ''.join(f'{res[c][v]["raw"]/res[c][v]["comp"]:15.2f}' for c in short))
print('\n=== per-variable compress MB/s (hard vars) ===')
print(f'{"var":34s} {"dtype":8s}' + ''.join(f'{s[:14]:>15s}' for s in short))
for v in hard:
    print(f'{v:34s} {meta[v]["enc_dtype"]:8s}' + ''.join(f'{res[c][v]["raw"]/res[c][v]["ct"]/1e6:15.0f}' for c in short))
print('\n=== per-variable decompress MB/s (hard vars) ===')
print(f'{"var":34s} {"dtype":8s}' + ''.join(f'{s[:14]:>15s}' for s in short))
for v in hard:
    print(f'{v:34s} {meta[v]["enc_dtype"]:8s}' + ''.join(f'{res[c][v]["raw"]/res[c][v]["dt"]/1e6:15.0f}' for c in short))
