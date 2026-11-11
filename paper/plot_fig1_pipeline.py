#!/usr/bin/env python3
"""
生成 Paper 1 图 1：Echo-QAOA 奇偶消耦响应设计、逐比特量子教师边缘蒸馏（Echo-Prior）、
零量子评估分布重叠解码与全角度能量重耦合总览图。
遵循 ICML 双栏出版规范（宽 175 mm，全部渲染字符 >= 5.0 pt，Computer Modern 标准数学字体，纯矢量零碰撞）。
"""

from pathlib import Path
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 7.5,
    "mathtext.fontset": "cm",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.dpi": 600,
    "figure.facecolor": "white",
})


def flat_top_magnetization(t: np.ndarray) -> np.ndarray:
    """计算 k=4 平顶 Echo 的精确单比特磁化响应 m_4(t) = 2 I_{(1+t)/2}(4, 4) - 1。"""
    q = 0.5 * (1.0 + np.clip(t, -1.0, 1.0))
    y = 1.0 - q
    p_k = (q ** 4) * (1.0 + 4.0 * y + 10.0 * (y ** 2) + 20.0 * (y ** 3))
    return 2.0 * p_k - 1.0


def create_fig1() -> None:
    """绘制并导出 Paper 1 图 1 架构与物理机制总览图。"""
    mm_to_inch = 1.0 / 25.4
    fig_w, fig_h = 175.0 * mm_to_inch, 86.0 * mm_to_inch
    aspect = fig_w / fig_h

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=600)
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 100.0)
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")

    # 统一克制的双色 + 中性灰体系（与图 2、图 3 色板同源）
    C_TEAL = "#007F78"       # 唯一核心主强调色：Echo-QAOA 流形与 Echo-Prior
    BG_TEAL = "#F2FBF9"
    BG_TEAL_SOFT = "#F7FCFB"
    C_SLATE = "#3B6998"      # 阶段二全角度量子重耦合
    BG_SLATE = "#F4F8FC"
    C_INK = "#0F172A"        # 统一深墨色主文字
    C_SUB = "#334155"        # 次级深灰文字
    C_BORDER = "#94A3B8"     # 中性边框
    BG_NEUTRAL = "#F8FAFC"   # 中性极浅底

    def add_box(x, y, w, h, fc="white", ec=C_BORDER, lw=0.85, ls="-", rounding=1.3, z=2):
        """添加统一圆角矩形框。"""
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.0,rounding_size={rounding}",
            facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z,
        )
        ax.add_patch(patch)
        return patch

    def add_circle(cx, cy, r_x, fc=C_TEAL, ec="none", lw=0.0, z=6):
        """按画布物理宽高比补偿绘制严格正圆。"""
        ell = Ellipse(
            (cx, cy), width=2.0 * r_x, height=2.0 * r_x * aspect,
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
        )
        ax.add_patch(ell)
        return ell

    def add_arrow(x1, y1, x2, y2, color=C_INK, lw=1.3, ls="-", scale=9.5, z=6):
        """添加直线有向箭头（零折弯）。"""
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                shrinkA=0.0, shrinkB=0.0, mutation_scale=scale,
            ),
            zorder=z,
        )

    def add_step_badge(cx, cy, num_str, color=C_TEAL, z=7):
        """在主干模块左上角绘制正圆步骤编号。"""
        add_circle(cx, cy, r_x=1.25, fc=color, z=z)
        ax.text(cx, cy, num_str, fontsize=7.2, fontweight="bold", color="white",
                ha="center", va="center", zorder=z + 1)

    # =========================================================================
    # 顶层左框 A：Offline Teacher Distillation (正对下方模块 2)
    # =========================================================================
    add_box(1.0, 64.5, 44.5, 33.8, fc=BG_NEUTRAL, ec=C_BORDER, lw=0.85, rounding=1.4, z=1)
    ax.text(2.6, 94.6, "A. Offline Teacher Distillation",
            fontsize=7.6, fontweight="bold", color=C_INK, ha="left", va="center", zorder=3)
    ax.text(44.0, 94.6, "4,096 graphs",
            fontsize=6.8, color=C_SUB, ha="right", va="center", zorder=3)

    add_box(2.2, 67.5, 10.8, 22.8, fc="white", ec=C_BORDER, lw=0.75, rounding=1.0, z=2)
    ax.text(7.6, 85.2, "Train Graph", fontsize=7.0, fontweight="bold", color=C_INK, ha="center", va="center", zorder=3)
    ax.text(7.6, 78.5, r"$(J, h, s^*)$", fontsize=7.8, color=C_INK, ha="center", va="center", zorder=3)
    ax.text(7.6, 71.8, "Known target", fontsize=6.5, color=C_SUB, ha="center", va="center", zorder=3)

    add_arrow(13.2, 78.8, 15.0, 78.8, color=C_SUB, lw=1.0, scale=8.0)

    add_box(15.2, 67.5, 12.2, 22.8, fc="white", ec=C_BORDER, lw=0.75, rounding=1.0, z=2)
    ax.text(21.3, 85.2, "Echo Teacher", fontsize=7.0, fontweight="bold", color=C_INK, ha="center", va="center", zorder=3)
    ax.text(21.3, 78.5, r"$|\psi_p(\vartheta^{\mathrm{T}})\rangle$", fontsize=7.8, color=C_INK, ha="center", va="center", zorder=3)
    ax.text(21.3, 71.8, "1D + L-BFGS-B", fontsize=6.5, color=C_SUB, ha="center", va="center", zorder=3)

    add_arrow(27.6, 78.8, 29.4, 78.8, color=C_TEAL, lw=1.0, scale=8.0)

    add_box(29.6, 67.5, 14.7, 22.8, fc=BG_TEAL_SOFT, ec=C_TEAL, lw=0.85, rounding=1.0, z=2)
    ax.text(36.95, 85.2, "Spin Marginals", fontsize=7.0, fontweight="bold", color=C_TEAL, ha="center", va="center", zorder=3)
    ax.text(36.95, 78.5, r"$\pi_i^{\mathrm{T}} = (1 + \langle Z_i \rangle_{\mathrm{T}})/2$",
            fontsize=7.6, color=C_INK, ha="center", va="center", zorder=3)
    ax.text(36.95, 71.8, "Smooth label", fontsize=6.5, color=C_SUB, ha="center", va="center", zorder=3)

    # =========================================================================
    # 顶层右框 B：Spin-Echo Parity Decoupling (正对下方模块 3)
    # =========================================================================
    add_box(48.5, 64.5, 50.5, 33.8, fc=BG_TEAL_SOFT, ec=C_TEAL, lw=0.85, rounding=1.4, z=1)
    ax.text(50.2, 94.6, "B. Spin-Echo Parity Decoupling",
            fontsize=7.6, fontweight="bold", color=C_TEAL, ha="left", va="center", zorder=3)
    ax.text(97.4, 94.6, "Propositions 1–4",
            fontsize=6.8, color=C_SUB, ha="right", va="center", zorder=3)

    # 绘制第 j 对完整 Echo 量子门线路示意
    y_q1, y_q2 = 85.8, 79.0
    ax.text(52.6, y_q1, r"$|+\rangle$", fontsize=7.6, color=C_INK, ha="right", va="center", zorder=3)
    ax.text(52.6, y_q2, r"$|+\rangle$", fontsize=7.6, color=C_INK, ha="right", va="center", zorder=3)
    ax.plot([53.2, 97.2], [y_q1, y_q1], color=C_SUB, lw=0.8, zorder=2)
    ax.plot([53.2, 97.2], [y_q2, y_q2], color=C_SUB, lw=0.8, zorder=2)

    gates = [
        (54.8, 8.8, r"$U_C(-g_j)$", "white", C_BORDER, C_INK),
        (65.2, 11.0, r"$U_B(\pi/2) \propto F$", "#E6F4F3", C_TEAL, C_TEAL),
        (77.8, 8.4, r"$U_C(g_j)$", "white", C_BORDER, C_INK),
        (87.8, 8.2, r"$U_B(\theta_j)$", "white", C_TEAL, C_INK),
    ]
    for gx, gw, gtxt, gfc, gec, gtc in gates:
        add_box(gx, 76.4, gw, 12.0, fc=gfc, ec=gec, lw=0.8, rounding=0.8, z=4)
        ax.text(gx + gw / 2.0, 82.4, gtxt, fontsize=7.6, color=gtc, ha="center", va="center", zorder=5)

    ax.text(
        73.75, 69.6,
        r"$F H_J F = H_J,\; F H_h F = -H_h \;\Rightarrow\; E(g_j, \theta_j) = U_B(\theta_j)\exp(-2ig_j H_h)F$",
        fontsize=7.6, color=C_INK, ha="center", va="center", zorder=3,
    )

    # =========================================================================
    # 中间连接走廊（y = 48.0 .. 64.5）：零遮挡垂直箭头 + 旁置清晰标注
    # =========================================================================
    ax.text(1.0, 56.4, "C. Online Deployment",
            fontsize=7.6, fontweight="bold", color=C_TEAL, ha="left", va="center", zorder=3)
    ax.text(1.0, 51.6, "Standard QAOA ansatz preserved",
            fontsize=6.7, color=C_SUB, ha="left", va="center", zorder=3)

    # A -> 模块 2 (x = 34.0，虚线唯一语义：离线冻结权重迁移)
    add_arrow(34.0, 64.2, 34.0, 48.4, color=C_TEAL, lw=1.25, ls="--", scale=9.0)
    ax.text(35.4, 56.4, "Frozen weights",
            fontsize=6.9, fontweight="bold", color=C_TEAL, ha="left", va="center", zorder=4)
    ax.text(35.4, 51.6, "(BCE distillation)",
            fontsize=6.6, color=C_SUB, ha="left", va="center", zorder=4)

    # B -> 模块 3 (x = 62.75，实线：精确单比特乘积响应注入解码器)
    add_arrow(62.75, 64.2, 62.75, 48.4, color=C_TEAL, lw=1.25, ls="-", scale=9.0)
    ax.text(64.2, 56.6, "Exact 1-qubit response",
            fontsize=6.9, fontweight="bold", color=C_INK, ha="left", va="center", zorder=4)
    ax.text(64.2, 51.6, r"$q_i(a) = [1 + m_k(\sin 4ah_i)]/2$",
            fontsize=7.6, color=C_INK, ha="left", va="center", zorder=4)

    # =========================================================================
    # 底层主干：ONLINE DEPLOYMENT SPINE (从左到右单调推进 1 -> 2 -> 3 -> 4)
    # =========================================================================
    box_y, box_h = 2.2, 45.8
    mid_y = box_y + 0.5 * box_h

    # --- Box 1: Unseen Instance ---
    b1_x, b1_w = 1.0, 18.0
    add_box(b1_x, box_y, b1_w, box_h, fc="white", ec=C_BORDER, lw=0.9, rounding=1.4, z=2)
    add_step_badge(b1_x + 2.2, box_y + box_h - 3.6, "1", color=C_SUB)
    ax.text(b1_x + 4.1, box_y + box_h - 3.6, "Unseen Instance",
            fontsize=7.1, fontweight="bold", color=C_INK, ha="left", va="center", zorder=4)

    nodes = [
        (5.5, 34.8, C_TEAL), (10.2, 37.6, C_SLATE), (14.5, 34.0, C_TEAL),
        (6.1, 25.8, "#64748B"), (10.9, 23.4, C_TEAL), (14.3, 26.8, C_SLATE),
    ]
    edges = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 5), (2, 5), (1, 4)]
    for u, v in edges:
        ax.plot([nodes[u][0], nodes[v][0]], [nodes[u][1], nodes[v][1]],
                color="#94A3B8", lw=0.95, zorder=3)
    for nx, ny, nc in nodes:
        add_circle(nx, ny, r_x=0.88, fc=nc, ec="white", lw=0.6, z=4)

    ax.text(b1_x + b1_w / 2.0, 16.6, r"Input $(J, h)$ only",
            fontsize=7.6, color=C_INK, ha="center", va="center", zorder=4)
    ax.text(b1_x + b1_w / 2.0, 11.2, r"Target $s^*$ unknown",
            fontsize=7.5, color=C_SUB, ha="center", va="center", zorder=4)
    ax.text(b1_x + b1_w / 2.0, 6.0, r"Any graph size $n$",
            fontsize=7.0, color=C_SUB, ha="center", va="center", zorder=4)

    # 主干粗箭头 1 -> 2
    add_arrow(19.2, mid_y, 22.3, mid_y, color=C_TEAL, lw=1.7, scale=10.5)

    # --- Box 2: Echo-Prior Encoder ---
    b2_x, b2_w = 22.5, 23.0
    add_box(b2_x, box_y, b2_w, box_h, fc=BG_TEAL, ec=C_TEAL, lw=1.15, rounding=1.4, z=2)
    add_step_badge(b2_x + 2.2, box_y + box_h - 3.6, "2", color=C_TEAL)
    ax.text(b2_x + 4.1, box_y + box_h - 3.6, "Echo-Prior Encoder",
            fontsize=7.1, fontweight="bold", color=C_INK, ha="left", va="center", zorder=4)

    ax.text(b2_x + b2_w / 2.0, 38.6, r"Equivariant GNN: $\hat{\pi} \in [0, 1]^n$",
            fontsize=7.5, color=C_INK, ha="center", va="center", zorder=4)

    bar_vals = [0.86, 0.22, 0.78, 0.16, 0.68]
    bar_y0 = 33.2
    for idx_b, val in enumerate(bar_vals):
        by = bar_y0 - idx_b * 3.8
        ax.text(b2_x + 4.7, by, rf"$\hat{{\pi}}_{{{idx_b + 1}}}$",
                fontsize=7.6, color=C_INK, ha="right", va="center", zorder=4)
        add_box(b2_x + 5.7, by - 1.05, 13.8, 2.1, fc="white", ec="#CBD5E1", lw=0.55, rounding=0.4, z=3)
        add_box(b2_x + 5.7, by - 1.05, 13.8 * val, 2.1, fc=C_TEAL, ec="none", lw=0.0, rounding=0.4, z=4)

    ax.text(b2_x + b2_w / 2.0, 11.2, "Node-level spin marginals",
            fontsize=6.9, color=C_INK, ha="center", va="center", zorder=4)
    ax.text(b2_x + b2_w / 2.0, 6.0, "No global angle jump",
            fontsize=6.7, color=C_SUB, ha="center", va="center", zorder=4)

    # 主干粗箭头 2 -> 3
    add_arrow(45.7, mid_y, 48.3, mid_y, color=C_TEAL, lw=1.7, scale=10.5)

    # --- Box 3: Flat-Top Overlap Decoder (Hero Module) ---
    b3_x, b3_w = 48.5, 28.5
    add_box(b3_x, box_y, b3_w, box_h, fc=BG_TEAL, ec=C_TEAL, lw=1.25, rounding=1.4, z=2)
    add_step_badge(b3_x + 2.2, box_y + box_h - 3.6, "3", color=C_TEAL)
    ax.text(b3_x + 4.1, box_y + box_h - 3.6, "Flat-Top Overlap Decoder",
            fontsize=7.1, fontweight="bold", color=C_INK, ha="left", va="center", zorder=4)

    ax.text(
        b3_x + b3_w / 2.0, 38.6,
        r"$S(a) = {\prod}_{i=1}^{n} [\hat{\pi}_i q_i(a) + (1 - \hat{\pi}_i)(1 - q_i(a))]$",
        fontsize=7.6, color=C_INK, ha="center", va="center", zorder=4,
    )

    # 嵌入式科学小图：Daubechies Flat-Top 响应 vs 正弦响应
    in_ax = ax.inset_axes([0.525, 0.190, 0.215, 0.145], zorder=4)
    phi = np.linspace(-np.pi, np.pi, 300)
    in_ax.axhline(0.0, color="#CBD5E1", lw=0.5, zorder=1)
    in_ax.plot(phi, np.sin(phi), color="#94A3B8", lw=0.9, ls="--")
    in_ax.plot(phi, flat_top_magnetization(np.sin(phi)), color=C_TEAL, lw=1.45)
    in_ax.set_xlim(-np.pi, np.pi)
    in_ax.set_ylim(-1.18, 1.28)
    in_ax.set_xticks([-np.pi / 2, 0.0, np.pi / 2])
    in_ax.set_xticklabels([r"$-\pi/2$", r"$0$", r"$\pi/2$"], fontsize=7.2, color=C_SUB)
    in_ax.set_yticks([-1, 1])
    in_ax.set_yticklabels([r"$-1$", r"$+1$"], fontsize=7.2, color=C_SUB)
    in_ax.tick_params(axis="both", length=1.8, width=0.55, pad=1.6)
    in_ax.spines["top"].set_visible(False)
    in_ax.spines["right"].set_visible(False)
    in_ax.spines["left"].set_linewidth(0.6)
    in_ax.spines["bottom"].set_linewidth(0.6)
    in_ax.spines["left"].set_color("#64748B")
    in_ax.spines["bottom"].set_color("#64748B")
    in_ax.text(np.pi / 2, 1.06, "Flat-Top", fontsize=6.6,
               color=C_TEAL, fontweight="bold", ha="center", va="bottom")

    ax.text(b3_x + b3_w / 2.0, 10.8,
            "0 quantum evals (128-pt grid)",
            fontsize=6.9, fontweight="bold", color=C_TEAL, ha="center", va="center", zorder=4)
    ax.text(b3_x + b3_w / 2.0, 5.8,
            r"$\hat{a} = \arg\max_{a}\, S(a) \;\mapsto\; \vartheta_0$",
            fontsize=7.6, color=C_INK, ha="center", va="center", zorder=4)

    # 主干粗箭头 3 -> 4
    add_arrow(77.2, mid_y, 79.8, mid_y, color=C_SLATE, lw=1.7, scale=10.5)

    # --- Box 4: Full-Angle Recoupling ---
    b4_x, b4_w = 80.0, 19.0
    add_box(b4_x, box_y, b4_w, box_h, fc=BG_SLATE, ec=C_SLATE, lw=1.05, rounding=1.4, z=2)
    add_step_badge(b4_x + 2.2, box_y + box_h - 3.6, "4", color=C_SLATE)
    ax.text(b4_x + 4.1, box_y + box_h - 3.6, "2p Recoupling",
            fontsize=7.1, fontweight="bold", color=C_INK, ha="left", va="center", zorder=4)

    ax.text(b4_x + b4_w / 2.0, 38.6,
            r"$\min_{\vartheta}\; \langle \psi_p(\vartheta) | H_C | \psi_p(\vartheta) \rangle$",
            fontsize=7.6, color=C_INK, ha="center", va="center", zorder=4)

    add_box(b4_x + 1.1, 21.2, b4_w - 2.2, 12.6, fc="white", ec="#93B4D4", lw=0.75, rounding=0.9, z=3)
    ax.text(b4_x + b4_w / 2.0, 30.0,
            r"$(-g_j + \delta_j,\, g_j + \delta_j)$",
            fontsize=7.5, color=C_INK, ha="center", va="center", zorder=4)
    ax.text(b4_x + b4_w / 2.0, 24.4,
            r"Restores $\exp(-2i\delta_j H_J)$",
            fontsize=7.2, color=C_SLATE, fontweight="bold", ha="center", va="center", zorder=4)

    ax.text(b4_x + b4_w / 2.0, 14.2,
            r"Releases all $2p$ angles",
            fontsize=7.0, color=C_SUB, ha="center", va="center", zorder=4)
    ax.text(b4_x + b4_w / 2.0, 6.8,
            r"Output $|\psi_p(\vartheta^*)\rangle$",
            fontsize=7.6, fontweight="bold", color=C_INK, ha="center", va="center", zorder=4)

    pdf_path = OUT / "fig1_echo_qaoa_overview.pdf"
    svg_path = OUT / "fig1_echo_qaoa_overview.svg"
    png_path = OUT / "fig1_echo_qaoa_overview.png"

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.savefig(png_path, format="png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    svg_content = svg_path.read_text(encoding="utf-8")
    svg_clean = re.sub(r"<!DOCTYPE[\s\S]*?>", "", svg_content)
    svg_path.write_text(svg_clean, encoding="utf-8")
    print(f"Generated Paper 1 Figure 1:\n  {pdf_path}\n  {svg_path}\n  {png_path}")


if __name__ == "__main__":
    create_fig1()
