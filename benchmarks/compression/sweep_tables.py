"""Markdown tables from a ``chunk_size_sweep --output-json`` file: decompress speed, compress
speed and whole-file size vs block size, one row per codec, identical columns in all three.

Usage:
    uv run python -m benchmarks.compression.sweep_tables SWEEP.json [--elements 2153952,268272,...]

Numbers use the same aggregation as ``chunk_size_sweep.summarise``: throughput pooled over
variables, size = each variable's full raw bytes x its sampled ratio at that block size, with
each variable contributing its nearest rung.
"""
import argparse
import ast
import json

import numpy as np


def nearest_rung(shapes, e):
    es = {int(np.prod(ast.literal_eval(s))): s for s in shapes}
    k = min(es, key=lambda x: abs(np.log(x) - np.log(e)))
    return es[k]


def fmt_elems(e):
    """Decimal labels, matching the README: 2153952 -> '2.15 M', 268272 -> '268 K', 4080 -> '4 K'."""
    if e >= 1_000_000:
        return f'{e / 1e6:.2f} M'
    if e >= 2000:
        return f'{e / 1000:.0f} K'
    return str(e)


def tables(res, elements=None):
    vs = res['vars']
    codecs = res['codecs']
    grid = sorted({int(np.prod(ast.literal_eval(s))) for r in vs.values() for s in r['shapes']}, reverse=True)
    if elements:
        grid = [e for e in grid if any(abs(np.log(e) - np.log(x)) < 0.05 for x in elements)]
    total_raw = sum(r['full_raw_bytes'] for r in vs.values())

    def cell(c, e, what):
        raw = t = proj = 0.0
        for r in vs.values():
            rr = r['shapes'][nearest_rung(r['shapes'], e)][c]
            raw += rr['raw']; t += rr[what] if what in ('ct', 'dt') else 0
            proj += r['full_raw_bytes'] * rr['comp'] / max(rr['raw'], 1)
        if what == 'size':
            return proj / 1e6
        return raw / max(t, 1e-12) / 1e6

    head = '| pipeline | ' + ' | '.join(fmt_elems(e) for e in grid) + ' |'
    rule = '|---|' + '---|' * len(grid)
    out = {}
    for what, title in (('dt', 'decompress MB/s'), ('ct', 'compress MB/s'), ('size', 'whole-file MB')):
        rows = [head, rule]
        for c in codecs:
            rows.append(f'| {c} | ' + ' | '.join(f'{cell(c, e, what):.0f}' for e in grid) + ' |')
        out[title] = '\n'.join(rows)
    out['raw_mb'] = total_raw / 1e6
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path')
    ap.add_argument('--elements', default=None, help='comma-separated block sizes (elements) to keep; default all rungs')
    a = ap.parse_args()
    res = json.load(open(a.path))
    t = tables(res, [int(x) for x in a.elements.split(',')] if a.elements else None)
    print(f'raw (encoded) bytes: {t.pop("raw_mb"):.1f} MB\n')
    for title, md in t.items():
        print(f'### {title}\n\n{md}\n')


if __name__ == '__main__':
    main()
