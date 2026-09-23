"""Figures for the docs page ``docs/concepts/compression-benchmarks.md``, drawn from the committed
result JSON files with the same aggregation as the README tables (``codec_bench.summary_rows``,
``sweep_tables.table_values``). Each figure is written twice, for the site's light and dark themes.

Usage (matplotlib is not a cfdb dependency):
    uv run --with matplotlib python -m benchmarks.compression.plot_docs [--out docs/assets/benchmarks]
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.compression.codec_bench import summary_rows
from benchmarks.compression.sweep_tables import table_values

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / 'results'
D01 = RESULTS / '2026-09-22_wrf_v50_12km_d01'
ERA5 = RESULTS / '2026-09-22_gabriele_3km_era5'
STATIONS = RESULTS / '2026-09-23_ecan_streamflow'

# benchmark pipeline id -> (label, colour, in cfdb)
PIPELINES = {
    'zstd-1': ('zstd', '#4c78a8', True),
    'shuffle+zstd-1': ('zstd_shuffle (default)', '#12a38f', True),
    'lz4-1': ('lz4', '#f58518', True),
    'shuffle+lz4-1': ('lz4_shuffle', '#e45756', True),
    'blosc1 shuffle zstd-1': ('blosc1 shuffle + zstd', '#9467bd', False),
    'shuffle+ydelta+zstd-1': ('shuffle + y-delta + zstd', '#8c6d31', False),
    'zstd-3': ('zstd, level 3', '#7f7f7f', False),
    'shuffle+zstd-3': ('zstd_shuffle, level 3', '#17becf', False),
}
CFDB_ORDER = ['zstd-1', 'shuffle+zstd-1', 'lz4-1', 'shuffle+lz4-1']
DTYPE_COLOURS = {'uint16': '#4c78a8', 'uint32': '#12a38f', 'uint8': '#f58518', 'float32': '#e45756'}

THEMES = {
    'light': dict(fg='#222222', grid='#dddddd', muted='#666666'),
    'dark': dict(fg='#e6e6e6', grid='#4a4a4a', muted='#a0a0a0'),
}


def load(path):
    with open(path) as f:
        return json.load(f)


def style(theme):
    t = THEMES[theme]
    plt.rcParams.update({
        'svg.hashsalt': 'cfdb-docs', 'svg.fonttype': 'none', 'font.size': 10,
        'text.color': t['fg'], 'axes.labelcolor': t['fg'], 'axes.edgecolor': t['muted'],
        'xtick.color': t['fg'], 'ytick.color': t['fg'], 'axes.grid': True, 'grid.color': t['grid'],
        'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False,
        'figure.facecolor': 'none', 'axes.facecolor': 'none', 'savefig.transparent': True,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    return t


def save(fig, out, name, theme):
    fig.savefig(out / f'{name}-{theme}.svg', bbox_inches='tight', metadata={'Date': None})
    plt.close(fig)


def fmt_elems(e):
    if e >= 1_000_000:
        return f'{e / 1e6:.1f} M'
    if e >= 1000:
        return f'{e / 1000:.0f} K'
    return str(e)


# label offsets (points) per pipeline in the size-vs-speed panels, (decompress, compress)
LABEL_OFFSETS = {
    'zstd-1': ((8, 2), (8, 2)),
    'shuffle+zstd-1': ((9, -4), (9, 2)),
    'lz4-1': ((-8, -14), (8, 2)),
    'shuffle+lz4-1': ((8, 2), (-10, 9)),
    'blosc1 shuffle zstd-1': ((8, 5), (-8, 8)),
    'shuffle+ydelta+zstd-1': ((8, -3), (8, -3)),
    'zstd-3': ((8, -3), (8, -3)),
}


def codec_tradeoff(out, theme):
    """Whole-file size vs decompress and compress speed, d01 at its stored 4.3 MB chunks."""
    t = style(theme)
    _, _, _, main_rows = summary_rows(load(D01 / 'codec_bench_lz4.json'))
    _, _, _, extra_rows = summary_rows(load(D01 / 'codec_bench.json'))
    points = [(r, True) for r in main_rows] + [(r, False) for r in extra_rows if r[0] in LABEL_OFFSETS and r[0] not in CFDB_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for k, (ax, idx, xlabel) in enumerate(((axes[0], 3, 'decompress speed (MB/s)'), (axes[1], 2, 'compress speed (MB/s)'))):
        for row, in_cfdb in points:
            label, colour, _ = PIPELINES[row[0]]
            x, y = row[idx], row[5]
            if in_cfdb:
                ax.scatter(x, y, s=90, color=colour, zorder=3)
            else:
                ax.scatter(x, y, s=70, facecolor='none', edgecolor=colour, linewidth=1.5, zorder=3)
            dx, dy = LABEL_OFFSETS[row[0]][k]
            ax.annotate(label, (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=8.5,
                        ha='right' if dx < 0 else 'left', va='center',
                        color=colour if in_cfdb else t['muted'])
        ax.set_xlabel(xlabel)
        ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.3)
    axes[0].set_ylabel('whole file (MB)')
    axes[0].set_ylim(0, 800)
    fig.suptitle('Size vs speed at the stored 4.3 MB chunks (WRF grid, 1.1 GB encoded)\n'
                 'filled: cfdb options; hollow: measured alternatives, not in cfdb', color=t['fg'], fontsize=10)
    save(fig, out, 'codec-tradeoff', theme)


def block_size(out, theme):
    """Whole-file size, decompress and compress speed vs elements per block (d01 sweep)."""
    t = style(theme)
    grid, raw_mb, values = table_values(load(D01 / 'chunk_size_sweep_all.json'))
    x = np.array(grid)
    panels = (('size', 'whole file (MB)'), ('dt', 'decompress speed (MB/s)'), ('ct', 'compress speed (MB/s)'))
    fig, grid_axes = plt.subplots(2, 2, figsize=(10, 7.6))
    axes = [grid_axes[0, 0], grid_axes[0, 1], grid_axes[1, 0]]
    legend_ax = grid_axes[1, 1]
    for ax, (what, ylabel) in zip(axes, panels):
        for c, vals in values[what].items():
            label, colour, in_cfdb = PIPELINES[c]
            ax.plot(x, vals, marker='o', markersize=3.5, color=colour, label=label,
                    linewidth=2.4 if c == 'shuffle+zstd-1' else 1.5, linestyle='-' if in_cfdb else '--')
        ax.set_xscale('log')
        ax.invert_xaxis()
        ax.set_xlabel('elements per chunk (log scale)')
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.axvspan(2**18 / 1.5, 2**18 * 1.5, color='#12a38f', alpha=0.12, linewidth=0)
    ticks = [2_000_000, 200_000, 20_000, 2_000, 200]
    for ax in axes:
        ax.set_xticks(ticks, [fmt_elems(v) for v in ticks])
        ax.minorticks_off()
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.axis('off')
    legend_ax.legend(handles, labels, loc='center', fontsize=10,
                     title='solid: cfdb options\ndashed: not in cfdb\nshaded: cfdb default chunk size',
                     title_fontsize=9)
    fig.suptitle(f'Chunk size: WRF grid, all 33 variables ({raw_mb:.0f} MB encoded), one sweep', color=t['fg'])
    fig.tight_layout()
    save(fig, out, 'block-size', theme)


def per_variable(out, theme):
    """zstd_shuffle file size relative to zstd, per d01 variable (stored chunks)."""
    t = style(theme)
    res = load(D01 / 'codec_bench.json')
    base, shuf = res['results']['zstd-1'], res['results']['shuffle+zstd-1']
    rows = sorted(((v, shuf[v]['comp'] / base[v]['comp'], res['vars'][v]['enc_dtype'], base[v]['raw'])
                   for v in base), key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8, 8.5))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[DTYPE_COLOURS[r[2]] for r in rows], height=0.7)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(1.0, color=t['fg'], linewidth=0.9)
    for yi, (v, ratio, _, _) in zip(y, rows):
        mb = (base[v]['comp'] / 1e6, shuf[v]['comp'] / 1e6)
        text = f'{mb[0]:.1f} to {mb[1]:.1f} MB' if mb[0] >= 0.05 else f'{mb[0] * 1e3:.0f} to {mb[1] * 1e3:.0f} KB'
        ax.annotate(text, (max(ratio, 1.0), yi), xytext=(5, 0), textcoords='offset points', va='center',
                    fontsize=7.5, color=t['muted'])
    ax.set_xlim(0, 1.35)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel('zstd_shuffle size ÷ zstd size (below 1 = smaller); right: zstd size to zstd_shuffle size')
    ax.grid(axis='y', visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in DTYPE_COLOURS.values()]
    ax.legend(handles, [f'stored as {d}' for d in DTYPE_COLOURS], loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8.5)
    ax.set_title('Per variable: WRF grid, stored 4.3 MB chunks', color=t['fg'])
    save(fig, out, 'per-variable', theme)


def datasets(out, theme):
    """Size relative to zstd for the three other cfdb options, on the three datasets."""
    t = style(theme)
    sets = (('WRF grid\n33 vars, packed ints\n4.3 MB chunks', [D01 / 'codec_bench_lz4.json']),
            ('ERA5 3 km grid\n4 vars, raw float32\n2.5 MB chunks', [ERA5 / 'codec_bench.json']),
            ('hourly streamflow\n141 stations, packed uint32\n100 KB chunks', [STATIONS / 'codec_bench.json']))
    options = ['shuffle+zstd-1', 'lz4-1', 'shuffle+lz4-1']
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.26
    for i, (_, files) in enumerate(sets):
        rows = {r[0]: r for f in files for r in summary_rows(load(f))[3]}
        base = rows['zstd-1'][5]
        for j, c in enumerate(options):
            xpos = i + (j - 1) * width
            label, colour, _ = PIPELINES[c]
            if c in rows:
                v = rows[c][5] / base
                ax.bar(xpos, v, width * 0.92, color=colour, label=label if i == 0 else None)
                ax.annotate(f'{v:.2f}', (xpos, v), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
            else:
                ax.annotate('not\nmeasured', (xpos, 0.05), ha='center', fontsize=7.5, color=t['muted'])
    ax.axhline(1.0, color=t['fg'], linewidth=0.9)
    ax.annotate('zstd = 1', (len(sets) - 0.55, 1.0), xytext=(0, 4), textcoords='offset points', fontsize=8, color=t['muted'])
    ax.set_xticks(range(len(sets)), [s[0] for s in sets], fontsize=8.5)
    ax.set_ylabel('file size ÷ zstd file size')
    ax.set_ylim(0, 1.6)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper left', fontsize=8.5, ncols=3)
    ax.set_title('Whole-file size on three datasets, relative to zstd', color=t['fg'])
    save(fig, out, 'datasets', theme)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(HERE.parent.parent / 'docs' / 'assets' / 'benchmarks'))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        for fig in (codec_tradeoff, block_size, per_variable, datasets):
            fig(out, theme)
    print(f'wrote {len(list(out.glob("*.svg")))} figures to {out}')


if __name__ == '__main__':
    main()
