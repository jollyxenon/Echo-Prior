#!/usr/bin/env python3
"""生成 Paper 1 (ICML) 的数据主图、分组总表与源数据 CSV（输出至 paper/paper1/figures/）。"""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from learning.aggregation import aggregate, interval

OUT = HERE / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
QA = ROOT / 'runs/canonical/figure-qa'
QA.mkdir(parents=True, exist_ok=True)
ARCHIVE = ROOT / 'runs/models_and_eval'
AGG_PATH = ARCHIVE / 'aggregate.json'
if not AGG_PATH.exists() or json.loads(AGG_PATH.read_text(encoding='utf-8')).get('bootstrap_replicates') != 4000:
    aggregate(ARCHIVE, replicates=4000, seed=0)
GROUPS = json.loads(AGG_PATH.read_text(encoding='utf-8'))['groups']
BENCH = json.loads((ROOT / 'runs/benchmark/benchmark-summary.json').read_text(encoding='utf-8'))
MANIFEST = json.loads((ARCHIVE / 'manifest.json').read_text(encoding='utf-8'))

METHODS = ['Flat-Top', 'Alternating-Phase', 'Endpoint-Matched', 'RI-LBFGSB', 'TQA-LBFGSB', 'INTERP-LBFGSB']
NAMES = ['Echo-Flat', 'Echo-Alt', 'Echo-End', 'RI', 'TQA', 'INTERP']
COLORS = ['#007F78', '#C48B29', '#8774AC', '#B8BCC2', '#4676A8', '#8E969D']
SELECTED = [
    ('Echo-Prior', 'Flat-Top'),
    ('Echo-Prior', 'Endpoint-Matched'),
    ('Echo-Prior', 'Alternating-Phase'),
    ('GNN', 'TQA-LBFGSB'),
    ('MLP', 'TQA-LBFGSB'),
]
DISPLAY = ['Echo-Prior-Flat', 'Echo-Prior-End', 'Echo-Prior-Alt', 'GNN-TQA', 'MLP-TQA']
PALETTE = ['#007F78', '#8774AC', '#C48B29', '#4676A8', '#8E969D']
FMTS = ['o-', '^-', 's-', 'v--', 'D--']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'savefig.dpi': 600,
    'figure.facecolor': 'white',
})


def write_csv(name, rows):
    """保存与图形相同统计单位的未舍入源数据。"""
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def label(ax, letter, title):
    """固定物理偏移放置面板编号与短标题。"""
    ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(-25, 13), textcoords='offset points',
                fontsize=10, fontweight='bold', annotation_clip=False)
    ax.set_title(title, loc='left', pad=13)


def save(fig, name):
    """输出可编辑矢量和高分辨率预览。"""
    fig.canvas.draw()
    fig.savefig(OUT / (name + '.pdf'))
    fig.savefig(OUT / (name + '.svg'))
    fig.savefig(OUT / (name + '.png'), dpi=600)
    plt.close(fig)


# =============================================================================
# 1. 五家族离线概率寻优主图 (benchmark-comparison) 与汇总表 (benchmark-tables)
# =============================================================================
BENCH_METHODS = ['Flat-Top', 'Alternating-Phase', 'Endpoint-Matched', 'TQA-LBFGSB', 'INTERP-LBFGSB', 'RI-LBFGSB']
BENCH_NAMES = ['Echo-Flat', 'Echo-Alt', 'Echo-End', 'TQA', 'INTERP', 'RI']
BENCH_COLORS = ['#007F78', '#C48B29', '#8774AC', '#4676A8', '#8E969D', '#B8BCC2']
BENCH_FMTS = ['o-', 's-', '^-', 'v-', 'P--', 'D--']

records = BENCH['per_instance_depth_method']
ids = sorted({r['instance'] for r in records})
depths = sorted({int(r['depth']) for r in records})
all_ns = sorted({int(r['n']) for r in records})

instance_values = {m: {i: [] for i in ids} for m in BENCH_METHODS}
for row in records:
    instance_values[row['method']][row['instance']].append(row['mean_pgs'])
write_csv('benchmark-per-instance.csv', [
    dict(id=instance, method=method, mean_percent=float(np.mean(instance_values[method][instance]) * 100))
    for instance in ids for method in BENCH_METHODS
])

bench_rows = []
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8))
fig.subplots_adjust(left=.125, right=.985, bottom=.22, top=.82, wspace=.48)
a, b, c = axes
label(a, 'a', 'Overall distribution')
label(b, 'b', f'Depth scaling (p = {depths[0]}..{depths[-1]})')
label(c, 'c', 'Family advantage vs TQA')

for i, (method, color) in enumerate(zip(BENCH_METHODS, BENCH_COLORS)):
    values = np.array([np.mean(instance_values[method][x]) for x in ids]) * 100
    mean, ci = interval(values)
    parts = a.violinplot(values, positions=[i], widths=.68, showextrema=False, orientation='horizontal')
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('none')
        pc.set_alpha(.30)
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    a.plot([q25, q75], [i, i], color=color, lw=2.2, solid_capstyle='round')
    a.errorbar(mean, i, xerr=[[mean - ci[0]], [ci[1] - mean]], fmt='o', color='white',
               markeredgecolor=color, markeredgewidth=1.3, ms=4.2, capsize=2, zorder=4)
    a.text(mean, i - .28, f'{mean:.1f}%', va='bottom', ha='center', fontsize=6.3,
           fontweight='bold' if i == 0 else 'normal', color=color)
    bench_rows.append(dict(method=method, mean=mean, ci_low=ci[0], ci_high=ci[1], unit='percent', instances=len(values)))
a.set(yticks=range(6), yticklabels=BENCH_NAMES, xlim=(-2, 106), xticks=[0, 25, 50, 75, 100],
      xlabel='Ground-state prob. (%)', ylim=(5.6, -.6))
a.grid(axis='x', color='#E8EBED', lw=.5)
a.set_axisbelow(True)

for i, (method, color, name, fmt) in enumerate(zip(BENCH_METHODS, BENCH_COLORS, BENCH_NAMES, BENCH_FMTS)):
    d_means, d_lows, d_highs = [], [], []
    for d in depths:
        sub = [r['mean_pgs'] for r in records if r['depth'] == d and r['method'] == method]
        m_val, ci = interval(np.array(sub) * 100)
        d_means.append(m_val)
        d_lows.append(m_val - ci[0])
        d_highs.append(ci[1] - m_val)
        bench_rows.append(dict(method=f'{method}/p{d}', mean=m_val, ci_low=ci[0], ci_high=ci[1],
                               unit='percent_at_depth', instances=len(sub)))
    b.errorbar(depths, d_means, yerr=[d_lows, d_highs], fmt=fmt,
               color=color, lw=1.3 if i == 0 else .85, ms=3.8, capsize=1.5, label=name)
b.set(xlim=(depths[0] - 0.7, depths[-1] + 0.7), xticks=depths, ylim=(15, 72), yticks=[20, 35, 50, 65],
      xlabel='Circuit depth p', ylabel='Mean probability (%)')
b.grid(axis='y', color='#E8EBED', lw=.5)
b.set_axisbelow(True)

families = ['dense_ising', 'sparse_ising', 'grid_ising', 'mwis', 'pinned_maxcut']
family_display = ['Dense', 'Sparse', 'Grid', 'MWIS', 'Pinned']
for i, (method, color) in enumerate(zip(BENCH_METHODS[:3], BENCH_COLORS[:3])):
    for j, family in enumerate(families):
        family_ids = [x for x in ids if x.startswith(family + '_')]
        delta = np.array([np.mean(instance_values[method][x]) - np.mean(instance_values['TQA-LBFGSB'][x]) for x in family_ids]) * 100
        mean, ci = interval(delta)
        c.errorbar(mean, j + (i - 1) * .22, xerr=[[mean - ci[0]], [ci[1] - mean]], fmt=['o', 's', '^'][i],
                   color=color, capsize=1.5, ms=3.8, lw=.8)
        bench_rows.append(dict(method=method + '/' + family, mean=mean, ci_low=ci[0], ci_high=ci[1],
                               unit='percentage_points_vs_TQA', instances=len(delta)))
c.axvline(0, color='#9BA3AA', lw=.7, ls='--')
c.set(yticks=range(5), yticklabels=family_display, ylim=(4.6, -.6), xlim=(-16, 22), xticks=[-10, 0, 10, 20],
      xlabel='Paired diff. vs TQA (pp)')
c.grid(axis='x', color='#E8EBED', lw=.5)
c.set_axisbelow(True)

fig.legend(handles=[Line2D([], [], marker=fmt[0], color=co, ls=fmt[1:], label=na)
                    for fmt, co, na in zip(BENCH_FMTS, BENCH_COLORS, BENCH_NAMES)],
           loc='lower center', bbox_to_anchor=(.53, .01), ncol=6, columnspacing=.8, fontsize=6.8, frameon=False)
save(fig, 'benchmark-comparison')
write_csv('benchmark-summary.csv', bench_rows)

benchmark_tables = {}
benchmark_table_rows = []
for grouping, key, levels in [
    ('overall', None, ['all']),
    ('by_family', 'family', families),
    ('by_depth', 'depth', depths),
    ('by_n', 'n', all_ns),
]:
    lines = []
    for level in levels:
        cells = []
        for method in METHODS:
            selected = [r for r in records if r['method'] == method and (key is None or r[key] == level)]
            grouped = {}
            for r in selected:
                grouped.setdefault(r['instance'], []).append(r['mean_pgs'])
            values = np.array([np.mean(grouped[i]) for i in sorted(grouped)]) * 100
            mean, ci = interval(values)
            cells.append((mean, ci))
            benchmark_table_rows.append(dict(grouping=grouping, level=level, method=method,
                                             mean_percent=mean, ci_low=ci[0], ci_high=ci[1], instances=len(values)))
        best = max(mean for mean, ci in cells)
        rendered = []
        for mean, ci in cells:
            central = r'\mathbf{' + f'{mean:.2f}' + '}' if mean == best else f'{mean:.2f}'
            rendered.append('$' + central + '^{+' + f'{ci[1] - mean:.2f}' + '}_{-' + f'{mean - ci[0]:.2f}' + '}$')
        lines.append((level, rendered))
    benchmark_tables[grouping] = lines
(OUT / 'benchmark-tables.json').write_text(json.dumps(benchmark_tables, indent=2), encoding='utf-8')
write_csv('benchmark-tables.csv', benchmark_table_rows)


# =============================================================================
# 2. MWIS n=10 神经参数生成主图 (neural-n10) 与分组表 (neural-n10-table)
# =============================================================================
data = {}
source_rows = []
for architecture, method in SELECTED:
    for depth in (8, 10, 12):
        jobs = [j for j in MANIFEST['jobs'] if j['n'] == 10 and j['depth'] == depth
                and j['architecture'] == architecture and j['method'] == method]
        initial, final = [], []
        for job in sorted(jobs, key=lambda j: j['seed']):
            rows = [json.loads(line) for line in (ARCHIVE / 'evaluation' / f'{job["id"]}.jsonl').read_text(encoding='utf-8').splitlines()]
            assert [r['id'] for r in rows] == MANIFEST['datasets']['10']['ids']
            initial.append([r['initial_probability'] for r in rows])
            final.append([r['probability'] for r in rows])
        before = np.mean(initial, axis=0) * 100
        after = np.mean(final, axis=0) * 100
        data[architecture, method, depth] = (before, after)
        for name, x, y in zip(MANIFEST['datasets']['10']['ids'], before, after):
            source_rows.append(dict(id=name, n=10, depth=depth, architecture=architecture, teacher=method,
                                    seed_count=len(jobs), initial_percent=x, final_percent=y))
write_csv('neural-n10-paired.csv', source_rows)

fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.6))
fig.subplots_adjust(left=.135, right=.975, bottom=.08, top=.91, wspace=.36, hspace=.40)
a, b, c, d = axes.flat
label(a, 'a', 'Initialization & energy refinement')
label(b, 'b', 'Final probability across depth')
label(c, 'c', 'All 1,024 test instances (ECDF)')
label(d, 'd', 'Paired advantage across depth')
stats = []

for i, ((architecture, method), name, color) in enumerate(zip(SELECTED, DISPLAY, PALETTE)):
    before = np.mean([data[architecture, method, p][0] for p in (8, 10, 12)], axis=0)
    after = np.mean([data[architecture, method, p][1] for p in (8, 10, 12)], axis=0)
    m0, ci0 = interval(before)
    m1, ci1 = interval(after)
    gain = m1 - m0
    a.plot([m0, m1], [i, i], color=color, lw=2.0, alpha=.85, zorder=2)
    a.errorbar(m0, i, xerr=[[m0 - ci0[0]], [ci0[1] - m0]], fmt='o', mfc='white', mec=color,
               mew=1.2, color=color, ms=5, capsize=2, zorder=3)
    a.errorbar(m1, i, xerr=[[m1 - ci1[0]], [ci1[1] - m1]], fmt='o', mfc=color, mec=color,
               color=color, ms=5.5, capsize=2, zorder=3)
    a.text(ci0[0] - 0.8, i, f'{m0:.1f}%', va='center', ha='right', fontsize=6.8, color='#5A646D')
    a.text(ci1[1] + 0.8, i, f'{m1:.1f}%', va='center', ha='left', fontsize=7.2,
           fontweight='bold' if i == 0 else 'normal', color=color)
    mid_x = (ci0[1] + ci1[0]) / 2
    a.text(mid_x, i - .27, f'+{gain:.1f} pp', va='bottom', ha='center', fontsize=6.3,
           fontweight='bold', color=color)
    stats.append(dict(model=name, depth='mean_8_10_12', initial=m0, final=m1, ci_low=ci1[0], ci_high=ci1[1]))

a.set(yticks=range(5), yticklabels=DISPLAY, ylim=(4.65, -.65), xlim=(-2, 34),
      xticks=[0, 10, 20, 30], xlabel='Ground-state probability (%)')

for yv in (10, 15, 20, 25, 30):
    b.plot([7.5, 12.18], [yv, yv], color='#E8EBED', lw=.5, zorder=0)
for i, ((architecture, method), name, color, fmt) in enumerate(zip(SELECTED, DISPLAY, PALETTE, FMTS)):
    means, lows, highs = [], [], []
    for p in (8, 10, 12):
        mean, ci = interval(data[architecture, method, p][1])
        means.append(mean)
        lows.append(mean - ci[0])
        highs.append(ci[1] - mean)
    x = np.array([8, 10, 12]) + (i - 2) * .08
    b.errorbar(x, means, yerr=[lows, highs], color=color, fmt=fmt,
               lw=1.3 if i == 0 else .9, ms=4.2, capsize=2, label=name)
    y_target = means[-1]
    if name == 'Echo-Prior-Flat':
        y_target = 30.4
    elif name == 'GNN-TQA':
        y_target = 28.4
    elif name == 'MLP-TQA':
        y_target = 26.3
    b.text(12.32, y_target, f'{means[-1]:.1f}%', va='center', fontsize=6.8,
           fontweight='bold' if i == 0 else 'normal', color=color)

b.set(xlim=(7.5, 13.2), xticks=[8, 10, 12], ylim=(6, 35), yticks=[10, 15, 20, 25, 30],
      xlabel='Circuit depth p', ylabel='Mean probability (%)')

p_flat = np.mean([data['Echo-Prior', 'Flat-Top', p][1] for p in (8, 10, 12)], axis=0)
p_gnn = np.mean([data['GNN', 'TQA-LBFGSB', p][1] for p in (8, 10, 12)], axis=0)
p_mlp = np.mean([data['MLP', 'TQA-LBFGSB', p][1] for p in (8, 10, 12)], axis=0)

for p_vals, col, name, ls, lw in [
    (p_flat, PALETTE[0], 'Echo-Prior-Flat', '-', 1.8),
    (p_gnn, PALETTE[3], 'GNN-TQA', '--', 1.2),
    (p_mlp, PALETTE[4], 'MLP-TQA', ':', 1.1),
]:
    sorted_v = np.sort(p_vals)
    cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
    c.step(sorted_v, cdf, color=col, ls=ls, lw=lw, where='post', label=name, zorder=3)

c.axvline(50, color='#9BA3AA', lw=.7, ls='--', zorder=1)
pct_flat_50 = np.mean(p_flat >= 50) * 100
pct_gnn_50 = np.mean(p_gnn >= 50) * 100
c.text(52, .36, f'Pgs ≥ 50%:\nEcho-Prior-Flat: {pct_flat_50:.1f}%\nGNN-TQA: {pct_gnn_50:.1f}%',
       va='bottom', ha='left', fontsize=6.2, color='#2D3748')

c.set(xlim=(0, 100), ylim=(0, 1.02), xticks=[0, 25, 50, 75, 100], yticks=[0, .25, .5, .75, 1.0],
      xlabel='Ground-state probability (%)', ylabel='Empirical CDF')
c.legend(loc='lower right', bbox_to_anchor=(.99, .02), frameon=False, fontsize=6.4)

for p in (8, 10, 12):
    diff_mlp = data['Echo-Prior', 'Flat-Top', p][1] - data['MLP', 'TQA-LBFGSB', p][1]
    diff_gnn = data['Echo-Prior', 'Flat-Top', p][1] - data['GNN', 'TQA-LBFGSB', p][1]
    for k, (diff, col, name) in enumerate([(diff_mlp, PALETTE[4], 'vs. MLP-TQA'),
                                           (diff_gnn, PALETTE[3], 'vs. GNN-TQA')]):
        pos = p + (k - .5) * .55
        clipped = np.clip(diff, -38, 65)
        parts = d.violinplot(clipped, positions=[pos], widths=.44, showextrema=False, orientation='vertical')
        for pc in parts['bodies']:
            pc.set_facecolor(col)
            pc.set_edgecolor('none')
            pc.set_alpha(.30)
        q25, q50, q75 = np.percentile(diff, [25, 50, 75])
        d.plot([pos, pos], [q25, q75], color=col, lw=2.2, solid_capstyle='round')
        m_val, ci = interval(diff)
        d.errorbar(pos, m_val, yerr=[[m_val - ci[0]], [ci[1] - m_val]], fmt='o', color='white',
                   markeredgecolor=col, markeredgewidth=1.2, ms=3.8, capsize=1.5, zorder=4)
        d.text(pos, 67, f'{m_val:+.1f}', va='bottom', ha='center', fontsize=6.0,
               fontweight='bold' if k == 1 else 'normal', color=col)
        stats.append(dict(model='Echo-Prior-Flat minus ' + DISPLAY[3 + k], depth=p, initial='',
                          final=m_val, ci_low=ci[0], ci_high=ci[1]))

d.axhline(0, color='#9BA3AA', lw=.7, ls='--')
d.set(xlim=(7.1, 12.9), xticks=[8, 10, 12], ylim=(-42, 92), yticks=[-20, 0, 20, 40, 60],
      xlabel='Circuit depth p', ylabel='Paired difference (pp)')
d.legend(handles=[
    Line2D([], [], color=PALETTE[4], lw=2.2, label='vs. MLP-TQA'),
    Line2D([], [], color=PALETTE[3], lw=2.2, label='vs. GNN-TQA'),
], loc='upper left', bbox_to_anchor=(.01, 1.01), ncol=2, columnspacing=.8, handlelength=1.4,
    frameon=False, fontsize=6.2)

fig.legend(handles=[Line2D([], [], marker=fmt[0], color=co, ls='', label=n) for fmt, co, n in zip(FMTS, PALETTE, DISPLAY)],
           loc='upper center', bbox_to_anchor=(.55, .995), ncol=5, columnspacing=.8, handletextpad=.4, frameon=False, fontsize=7)
save(fig, 'neural-n10')
write_csv('neural-n10-summary.csv', stats)

# 结构化 15 类神经模型分组表 (neural-n10-table)
TEACHER_ORDER = ['TQA-LBFGSB', 'Flat-Top', 'Endpoint-Matched', 'Alternating-Phase', 'RI-LBFGSB', 'INTERP-LBFGSB']
row_groups = [
    ('Bitwise Marginal Prior (Proposed)', [
        ('Echo-Prior', 'Flat-Top'),
        ('Echo-Prior', 'Endpoint-Matched'),
        ('Echo-Prior', 'Alternating-Phase'),
    ]),
    ('Direct 2p-Angle Regression — GNN Backbone', [
        ('GNN', t) for t in TEACHER_ORDER
    ]),
    ('Direct 2p-Angle Regression — MLP Backbone', [
        ('MLP', t) for t in TEACHER_ORDER
    ]),
]
lookup = {(r['architecture'], r['method'], r['depth']): r for r in GROUPS if r['n'] == 10}

fig, ax = plt.subplots(figsize=(7.1, 5.5))
fig.subplots_adjust(left=.015, right=.985, bottom=.06, top=.92)
ax.set(xlim=(0, 1), ylim=(-.6, 18.2))
ax.axis('off')
ax.text(.015, 17.6, 'Model paradigm / Architecture', fontsize=8.5, fontweight='bold', color='#1A202C')
xcols = [.36, .54, .72]
for x, p in zip(xcols, (8, 10, 12)):
    ax.text(x, 17.6, f'p = {p}', ha='center', fontsize=9.5, fontweight='bold', color='#1A202C')
ax.text(.91, 17.6, 'Mean', ha='center', fontsize=9.5, fontweight='bold', color='#1A202C')
ax.axhline(17.15, color='#4A5568', lw=1.0)

rows = []
latex = []
y = 16.6
for group_name, specs in row_groups:
    ax.text(.015, y, group_name.upper(), fontsize=7.0, fontweight='bold', color='#4A5568', va='center')
    ax.axhline(y - .35, color='#CBD5E0', lw=.5)
    latex.append(r'\multicolumn{5}{l}{\textbf{' + group_name + r'}} \\ \midrule')
    y -= .75
    for architecture, method in specs:
        prefix = {'Echo-Prior': 'Echo-Prior', 'MLP': 'MLP', 'GNN': 'GNN'}[architecture]
        t_name = {
            'Flat-Top': 'Flat', 'Alternating-Phase': 'Alt', 'Endpoint-Matched': 'End',
            'RI-LBFGSB': 'RI', 'TQA-LBFGSB': 'TQA', 'INTERP-LBFGSB': 'INTERP',
        }[method]
        label_text = f'{prefix}-{t_name}'
        is_prior_flat = (architecture == 'Echo-Prior' and method == 'Flat-Top')
        is_prior_group = (architecture == 'Echo-Prior')

        if is_prior_group:
            ax.axhspan(y - .40, y + .40, color='#F0FDF4' if is_prior_flat else '#F7FAFC', zorder=0)

        name_color = '#007F78' if is_prior_flat else '#2D3748'
        ax.text(.025, y, label_text, va='center', fontsize=7.8,
                fontweight='bold' if is_prior_flat else 'normal', color=name_color)

        texcells = []
        p_means = []
        for x, p in zip(xcols, (8, 10, 12)):
            r = lookup[architecture, method, p]
            mean = r['mean_probability'] * 100
            init = r['mean_initial_probability'] * 100
            lo, hi = np.array(r['ci95']) * 100
            best = max(z['mean_probability'] for z in GROUPS if z['n'] == 10 and z['depth'] == p)
            bold = r['mean_probability'] == best
            central = (r'\mathbf{' + f'{mean:.2f}' + '}') if bold else f'{mean:.2f}'
            formula = '$' + central + '^{+' + f'{hi - mean:.2f}' + '}_{-' + f'{mean - lo:.2f}' + '}$'
            p_means.append(mean)

            ax.text(x - .065, y, f'{init:.1f} →', ha='right', va='center', fontsize=6.8, color='#718096')
            val_color = '#007F78' if bold else '#2D3748'
            ax.text(x - .005, y, f'{mean:.2f}', ha='center', va='center', fontsize=8.8,
                    fontweight='bold' if bold else 'normal', color=val_color)
            ax.text(x + .028, y + .18, f'+{hi - mean:.2f}', ha='left', va='center', fontsize=5.6, color=val_color)
            ax.text(x + .028, y - .18, f'−{mean - lo:.2f}', ha='left', va='center', fontsize=5.6, color=val_color)
            texcells.append(f'{init:.1f} $\\rightarrow$ {formula}')
            rows.append(dict(architecture=architecture, teacher=method, depth=p, initial_percent=init,
                             final_percent=mean, ci_low=lo, ci_high=hi, best_in_column=bold))

        cross_mean = np.mean(p_means)
        ax.text(.91, y, f'{cross_mean:.2f}%', ha='center', va='center', fontsize=8.5,
                fontweight='bold' if is_prior_flat else 'normal',
                color='#007F78' if is_prior_flat else '#2D3748')
        tex_mean = (r'$\mathbf{' + f'{cross_mean:.2f}' + r'}\%$') if is_prior_flat else f'${cross_mean:.2f}\\%$'
        texcells.append(tex_mean)
        latex.append(f'{label_text} & ' + ' & '.join(texcells) + r' \\')
        y -= .88
    y -= .22

ax.axhline(-.15, color='#4A5568', lw=1.0)
ax.text(.015, -.45, 'Values report initial → energy-refined ground-state probability (%) on 1,024 held-out MWIS test instances (n = 10, seed = 0; 95% bootstrap CI).',
        fontsize=6.3, color='#4A5568')
save(fig, 'neural-n10-table')
write_csv('neural-n10-table.csv', rows)
(OUT / 'neural-n10-table.tex').write_text(
    '\\begin{tabular}{lcccc}\n\\toprule\nModel & $p=8$ & $p=10$ & $p=12$ & Mean \\\\ \\midrule\n'
    + '\n'.join(latex) + '\n\\bottomrule\n\\end{tabular}\n',
    encoding='utf-8',
)
print("Successfully generated Paper 1 figures and tables in", OUT)
