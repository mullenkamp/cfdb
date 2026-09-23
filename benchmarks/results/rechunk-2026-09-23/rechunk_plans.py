"""Rechunker read plans on full-size real shapes, per default chunk target. No data needed:
rechunkit.calc_n_reads_rechunker gives (chunk reads, writes); each read decompresses one whole chunk."""
import math, rechunkit
MAXMEM = 2**29
TARGETS = [2**18, 2**19, 2**20, 2**21, 2**22]
cases = {
  'SST daily grid (16650,1000,1600) u16, 53 GB': ((16650, 1000, 1600), 2, {
      'one point, all 45 years':       ((16650, 1, 1), (slice(None), slice(500, 501), slice(800, 801))),
      'one day, full grid':            ((1, 1000, 1600), (slice(16000, 16001), slice(None), slice(None))),
      '100x100 region, all years':     ((16650, 100, 100), (slice(None), slice(400, 500), slice(700, 800))),
      'whole array -> daily grids':    ((1, 1000, 1600), None),
      'whole array -> monthly groups': ((30, 1000, 1600), None),
      'whole array -> pixel series':   ((16650, 10, 10), None)}),
  'forecast (29,49,14,165,221) u16': ((29, 49, 14, 165, 221), 2, {
      'one point, all runs/leads/levels': ((29, 49, 14, 1, 1), (slice(None), slice(None), slice(None), slice(80, 81), slice(110, 111))),
      'latest run, all fields':           ((1, 49, 14, 165, 221), (slice(28, 29), slice(None), slice(None), slice(None), slice(None))),
      'whole -> per lead time grids':     ((29, 1, 14, 165, 221), None)}),
  'stations ts_ortho (141,183227) u32': ((141, 183227), 4, {
      'one station, full history':    ((1, 183227), (slice(70, 71), slice(None))),
      'latest week, all stations':    ((141, 168), (slice(None), slice(183227 - 168, None))),
      'whole -> annual blocks':       ((141, 8760), None)}),
}
def explicit(sel, shape):
    if sel is None:
        return None
    return tuple(slice(0 if s.start is None else s.start, sh if s.stop is None else s.stop) for s, sh in zip(sel, shape))

for title, (shape, it, queries) in cases.items():
    print(f'\n### {title}')
    chunks = {t: rechunkit.guess_chunk_shape(shape, it, t) for t in TARGETS}
    print(f'{"target":>8s}: ' + ' | '.join(f'{t//1024:>5d} KiB {str(c):>22s} {math.prod(c)*it/1e6:5.2f}MB' for t, c in chunks.items()))
    for q, (tgt, sel) in queries.items():
        sel = explicit(sel, shape)
        out_bytes = math.prod(s.stop - s.start for s in sel) * it if sel else math.prod(shape) * it
        cells = []
        for t, c in chunks.items():
            n_reads, n_writes = rechunkit.calc_n_reads_rechunker(shape, it, c, tgt, MAXMEM, sel=sel)
            read_b = n_reads * math.prod(c) * it
            cells.append(f'{n_reads:>9d} reads {read_b/1e9:8.2f} GB {read_b/out_bytes:7.1f}x')
        print(f'  {q:32s} ' + ' | '.join(cells))
