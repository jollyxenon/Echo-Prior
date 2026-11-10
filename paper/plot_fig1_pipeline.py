#!/usr/bin/env python3
"""
生成 Paper 1 图 1：Echo-QAOA 的构造性响应设计、逐比特量子教师边缘蒸馏（Echo-Prior）、解析分布重叠解码与全角度能量重耦合流程图。
严格遵循 ICML 顶会双栏出版规范（宽度 183 mm，最小文字字号 >= 5.0 pt，纯矢量排版）。
"""

from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 7.2,
    'mathtext.fontset': 'dejavusans',
    'mathtext.default': 'regular',
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'savefig.dpi': 600,
    'figure.facecolor': 'white'
})


def create_fig1():
    """绘制并导出 Paper 1 图 1 架构总览图。"""
    mm_to_inch = 1.0 / 25.4
    fig_w, fig_h = 183.0 * mm_to_inch, 112.0 * mm_to_inch
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=600)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    C_TEAL = '#007F78'
    C_AMBER = '#B45309'
    C_PURPLE = '#6D28D9'
    C_DARK = '#0F172A'
    C_MUTED = '#334155'
    C_BORDER = '#94A3B8'
    C_BG_TOP = '#F8FAFC'
    C_BG_BOT = '#F0FDF4'
    BORDER_TOP = '#CBD5E1'
    BORDER_BOT = '#A7F3D0'

    rect_top = FancyBboxPatch((1.2, 53.5), 97.6, 44.5, boxstyle="round,pad=0.4,rounding_size=1.5",
                              facecolor=C_BG_TOP, edgecolor=BORDER_TOP, linewidth=0.8, zorder=1)
    ax.add_patch(rect_top)
    ax.text(3.5, 95.2, 'OFFLINE PHASE: Multi-Body Teacher Collection & Prior Learning',
            fontsize=8.5, fontweight='bold', color=C_DARK, zorder=2)

    rect_bot = FancyBboxPatch((1.2, 2.5), 97.6, 45.5, boxstyle="round,pad=0.4,rounding_size=1.5",
                              facecolor=C_BG_BOT, edgecolor=BORDER_BOT, linewidth=0.8, zorder=1)
    ax.add_patch(rect_bot)
    ax.text(3.5, 45.2, 'ONLINE DEPLOYMENT', fontsize=8.5, fontweight='bold', color=C_TEAL, zorder=2)
    ax.text(27.5, 45.2, 'Zero-Quantum Overlap Decoding & Full-Angle Energy Re-coupling',
            fontsize=8.0, fontweight='bold', color=C_DARK, zorder=2)

    def draw_card(x, y, w, h, items, border=C_BORDER, bg='white', z=3):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                             facecolor=bg, edgecolor=border, linewidth=0.8, zorder=z)
        ax.add_patch(box)
        n_items = len(items)
        pad_top = 3.2
        pad_bot = 2.8
        avail_h = h - pad_top - pad_bot
        step = avail_h / max(1, n_items - 0.2)
        y_start = y + h - pad_top - 0.4 * step

        for idx, item in enumerate(items):
            curr_y = y_start - idx * step
            itype = item.get('type', 'text')
            text = item.get('text', '')
            if itype == 'title':
                f_size = item.get('size', 7.8)
                ax.text(x + w / 2, curr_y, text, fontsize=f_size, fontweight='bold',
                        ha='center', va='center', color=C_DARK, zorder=z + 1)
            elif itype == 'subtitle':
                color = item.get('color', C_DARK)
                f_size = item.get('size', 7.2)
                ax.text(x + w / 2, curr_y, text, fontsize=f_size, fontstyle='italic',
                        ha='center', va='center', color=color, zorder=z + 1)
            elif itype == 'badge':
                b_color = item.get('color', C_TEAL)
                b_w = item.get('w', min(w - 3.0, 15.0))
                bbox = FancyBboxPatch((x + w / 2 - b_w / 2, curr_y - 1.2), b_w, 2.6,
                                      boxstyle="round,pad=0.1,rounding_size=0.6",
                                      facecolor=b_color, edgecolor='none', zorder=z + 1)
                ax.add_patch(bbox)
                ax.text(x + w / 2, curr_y, text, fontsize=6.8, fontweight='bold',
                        ha='center', va='center', color='white', zorder=z + 2)
            else:
                ax.text(x + w / 2, curr_y, text, fontsize=7.0,
                        ha='center', va='center', color=C_MUTED, zorder=z + 1)

    draw_card(3.5, 56.5, 20.5, 35.5, [
        {'type': 'title', 'text': 'Training Instances'},
        {'type': 'subtitle', 'text': 'Ising Problem (J, h)'},
        {'type': 'text', 'text': 'Known ground state s*'},
        {'type': 'text', 'text': '5 problem families'},
        {'type': 'text', 'text': '2,750 benchmark inst.'},
    ], border='#94A3B8')

    draw_card(28.0, 56.5, 22.0, 35.5, [
        {'type': 'title', 'text': 'Teacher Optimization'},
        {'type': 'subtitle', 'text': 'Echo 1D + L-BFGS-B'},
        {'type': 'text', 'text': 'Maximize prob P(s*)'},
        {'type': 'text', 'text': '10s exploration budget'},
        {'type': 'text', 'text': '6 distinct teacher pools'},
    ], border=C_PURPLE)

    draw_card(54.0, 56.5, 21.0, 35.5, [
        {'type': 'title', 'text': 'Teacher Marginals'},
        {'type': 'subtitle', 'text': 'πᵢᵀ = ⟨Zᵢ + 1⟩ / 2', 'color': C_PURPLE},
        {'type': 'badge', 'text': 'ENTANGLED STATE', 'color': C_PURPLE, 'w': 16.0},
        {'type': 'text', 'text': 'Multi-body quantum circuit'},
        {'type': 'text', 'text': 'Extract single-spin prob'},
        {'type': 'text', 'text': 'Supervises Echo-Prior encoder'},
    ], border=C_PURPLE)

    draw_card(79.0, 56.5, 18.5, 35.5, [
        {'type': 'title', 'text': 'Echo-Prior Encoder'},
        {'type': 'subtitle', 'text': 'π̂ = Echo-Prior(J, h)', 'color': C_TEAL},
        {'type': 'badge', 'text': 'SUPERVISED GNN', 'color': C_TEAL, 'w': 15.0},
        {'type': 'text', 'text': 'Permutation-equivariant'},
        {'type': 'text', 'text': 'Binary cross-entropy loss'},
        {'type': 'text', 'text': 'Trained on 4,096 graphs'},
    ], border=C_TEAL)

    draw_card(3.5, 5.5, 19.5, 36.5, [
        {'type': 'title', 'text': 'Unseen Instance'},
        {'type': 'subtitle', 'text': 'Input (J, h) Only'},
        {'type': 'text', 'text': 'Target state unknown'},
        {'type': 'text', 'text': 'No quantum oracle'},
        {'type': 'text', 'text': 'Pure classical input'},
        {'type': 'text', 'text': 'Arbitrary graph topology'},
    ], border='#94A3B8')

    draw_card(27.5, 5.5, 22.5, 36.5, [
        {'type': 'title', 'text': 'Echo-Prior Inference', 'size': 7.6},
        {'type': 'subtitle', 'text': 'π̂ = Echo-Prior(J, h)', 'color': C_TEAL, 'size': 7.2},
        {'type': 'badge', 'text': 'EQUIVARIANT INFERENCE', 'color': C_TEAL, 'w': 19.8},
        {'type': 'text', 'text': 'Amortized pass: O(L|E|)'},
        {'type': 'text', 'text': 'Smooth spin marginals ∈ [0,1]ⁿ'},
        {'type': 'text', 'text': 'Distilled multi-body bias'},
    ], border=C_TEAL, bg='#F0FDF4')

    draw_card(54.0, 5.5, 21.0, 36.5, [
        {'type': 'title', 'text': 'Flat-Top Decoder'},
        {'type': 'subtitle', 'text': 'maxₐ S(a) = ∏ᵢ [...]', 'color': C_TEAL},
        {'type': 'badge', 'text': '0 QUANTUM EVALS', 'color': C_TEAL, 'w': 16.0},
        {'type': 'text', 'text': '128-pt classical scalar grid'},
        {'type': 'text', 'text': 'Exact product overlap sum'},
        {'type': 'text', 'text': 'Decodes global angle: ϑ₀'},
        {'type': 'text', 'text': 'Runtime: O(Mnk) ≪ 2ⁿ'},
    ], border=C_TEAL)

    draw_card(78.5, 5.5, 19.0, 36.5, [
        {'type': 'title', 'text': '2p Energy Re-coupling', 'size': 7.4},
        {'type': 'subtitle', 'text': 'min_ϑ ⟨ψₚ(ϑ)| H_C |ψₚ(ϑ)⟩', 'color': C_AMBER, 'size': 6.8},
        {'type': 'badge', 'text': '2p RE-COUPLING', 'color': C_AMBER, 'w': 15.0},
        {'type': 'text', 'text': 'Unsupervised L-BFGS-B'},
        {'type': 'text', 'text': 'Breaks Echo anti-symmetry'},
        {'type': 'text', 'text': 'Restores H_J entanglement'},
        {'type': 'text', 'text': 'High ground-state prob.'},
    ], border=C_AMBER)

    def draw_arrow(x1, y1, x2, y2, color=C_DARK, lw=1.1, style='->', ls='-'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls,
                                    shrinkA=2, shrinkB=2, mutation_scale=10), zorder=6)

    draw_arrow(24.0, 74.2, 28.0, 74.2, color=C_DARK, lw=1.2)
    draw_arrow(50.0, 74.2, 54.0, 74.2, color=C_PURPLE, lw=1.2)
    draw_arrow(75.0, 74.2, 79.0, 74.2, color=C_TEAL, lw=1.2)

    draw_arrow(23.0, 23.8, 27.5, 23.8, color=C_TEAL, lw=1.2)
    draw_arrow(50.0, 23.8, 54.0, 23.8, color=C_TEAL, lw=1.2)
    draw_arrow(75.0, 23.8, 78.5, 23.8, color=C_AMBER, lw=1.3)

    corridor_x = [88.2, 88.2, 38.8, 38.8]
    corridor_y = [56.5, 50.8, 50.8, 42.0]
    ax.plot(corridor_x[:-1], corridor_y[:-1], color=C_TEAL, lw=1.1, linestyle='--', zorder=6)
    ax.annotate('', xy=(corridor_x[-1], corridor_y[-1]), xytext=(corridor_x[-2], corridor_y[-2]),
                arrowprops=dict(arrowstyle='->', color=C_TEAL, lw=1.1, linestyle='--',
                                shrinkA=0, shrinkB=2, mutation_scale=9), zorder=6)

    ax.text(63.5, 50.8, 'Frozen Weights Deployment', fontsize=7.0, ha='center', va='center',
            color=C_TEAL, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=C_TEAL, lw=0.6), zorder=7)

    plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)

    pdf_path = OUT / 'fig1_echo_qaoa_overview.pdf'
    svg_path = OUT / 'fig1_echo_qaoa_overview.svg'
    png_path = OUT / 'fig1_echo_qaoa_overview.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    svg_content = svg_path.read_text(encoding='utf-8')
    svg_clean = re.sub(r'<!DOCTYPE[\s\S]*?>', '', svg_content)
    svg_path.write_text(svg_clean, encoding='utf-8')
    print(f"Generated Paper 1 Figure 1:\n  {pdf_path}\n  {svg_path}\n  {png_path}")


if __name__ == '__main__':
    create_fig1()
