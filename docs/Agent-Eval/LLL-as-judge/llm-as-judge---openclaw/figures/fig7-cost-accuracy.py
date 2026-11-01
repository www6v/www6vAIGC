#!/usr/bin/env python3
"""
Fig 7: Cost vs Accuracy trade-off for LLM-as-a-Judge tiers.
Pareto frontier with "sweet spot" annotation.
Publication-quality style (DDIA-inspired clean design).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle
import os

# ── Data ──────────────────────────────────────────────────────────────
models = [
    {"name": "Human annotator",  "cost": 0.10,   "agree": 100.0, "color": "#D32F2F", "marker": "s", "s": 180, "z": 10},
    {"name": "GPT-4o",           "cost": 0.01,   "agree": 90.0,  "color": "#1565C0", "marker": "o", "s": 150, "z": 9},
    {"name": "Claude Sonnet",    "cost": 0.005,  "agree": 87.0,  "color": "#2E7D32", "marker": "^", "s": 140, "z": 8},
    {"name": "GPT-4o-mini",      "cost": 0.001,  "agree": 82.0,  "color": "#6A1B9A", "marker": "D", "s": 130, "z": 7},
    {"name": "Prometheus (7B)",  "cost": 0.0003, "agree": 78.0,  "color": "#E65100", "marker": "v", "s": 120, "z": 6},
    {"name": "Llama 3.1 8B",     "cost": 0.0002, "agree": 75.0,  "color": "#00838F", "marker": "*", "s": 160, "z": 5},
]

# ── Font fallback for Chinese ─────────────────────────────────────────
import matplotlib.font_manager as fm
chinese_fonts = [
    "Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei", "Source Han Sans SC", "SimHei",
    "Microsoft YaHei", "PingFang SC", "AR PL UMing CN",
    "Droid Sans Fallback", "AR PL UKai CN",
]
available = {f.name for f in fm.fontManager.ttflist}
chosen_cn = None
for f in chinese_fonts:
    if f in available:
        chosen_cn = f
        break
plt.rcParams["font.sans-serif"] = [chosen_cn, "DejaVu Sans", "Liberation Sans", "Arial"] if chosen_cn else ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── Figure ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6.2))
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FAFAFA")

# ── Plot points ───────────────────────────────────────────────────────
for m in models:
    ax.scatter(
        m["cost"], m["agree"],
        c=m["color"], marker=m["marker"], s=m["s"],
        zorder=m["z"], edgecolors="white", linewidths=1.2,
        label=m["name"],
    )
    # Name labels next to points
    dx = 1.08 if m["cost"] >= 0.005 else 1.12
    ax.annotate(
        m["name"],
        xy=(m["cost"], m["agree"]),
        xytext=(m["cost"] * dx, m["agree"] + (0.3 if m["agree"] < 85 else 0.5)),
        fontsize=8.5,
        fontweight="medium",
        color="#333333",
        ha="left",
        va="bottom",
    )

# ── Pareto frontier (smooth curve through frontier points) ───────────
# Frontier: Human → GPT-4o → Claude Sonnet → GPT-4o-mini → Prometheus
pareto_costs  = [0.10, 0.01, 0.005, 0.001, 0.0003]
pareto_agree  = [100, 90, 87, 82, 78]

# Fit a smooth interpolation using log(cost)
log_costs = np.log10(pareto_costs)
# Use numpy polynomial fit for smooth curve (no scipy dependency)
log_costs = np.array(log_costs)
pareto_agree = np.array(pareto_agree)
t_fine = np.linspace(min(log_costs), max(log_costs), 300)
coeffs = np.polyfit(log_costs, pareto_agree, 3)
agree_fine = np.polyval(coeffs, t_fine)

ax.plot(
    10 ** t_fine, agree_fine,
    color="#555555", linewidth=2.0, linestyle="--",
    alpha=0.7, zorder=3,
    label="Pareto 前沿 (Pareto frontier)",
)

# ── Sweet spot zone ───────────────────────────────────────────────────
# The "sweet spot" is between GPT-4o-mini and Claude Sonnet
sweet_rect = FancyBboxPatch(
    (0.0008, 79.5), 0.008, 10,
    boxstyle="round,pad=0.3",
    facecolor="#FFF9C4", alpha=0.45,
    edgecolor="#F9A825", linewidth=1.5, linestyle="-.",
    zorder=2,
)
ax.add_patch(sweet_rect)
ax.text(
    0.004, 88,
    "★ 性价比甜点区\n(Sweet Spot)",
    fontsize=9.5, fontweight="bold",
    ha="center", va="center",
    color="#E65100",
    zorder=4,
)

# ── Axes ──────────────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_xlim(0.0001, 0.2)
ax.set_ylim(70, 105)

ax.set_xlabel("单次评估成本 (USD)", fontsize=12, fontweight="medium")
ax.set_ylabel("与人工评估一致性 (Agreement %)", fontsize=12, fontweight="medium")

# Grid
ax.grid(True, which="major", alpha=0.25, color="#999999", linestyle="-")
ax.grid(True, which="minor", alpha=0.12, color="#BBBBBB", linestyle="--")

# Custom x-ticks with dollar formatting
xticks = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
ax.set_xticks(xticks)
ax.set_xticklabels([f"${x:.4f}" if x < 0.01 else f"${x:.2f}" for x in xticks], fontsize=9)

# ── Legend ────────────────────────────────────────────────────────────
ax.legend(
    loc="lower right", fontsize=8, framealpha=0.85,
    edgecolor="#CCCCCC", facecolor="#FFFFFF",
)

# ── Title ─────────────────────────────────────────────────────────────
ax.set_title(
    "LLM-as-a-Judge: 成本 vs 评估一致性权衡",
    fontsize=14, fontweight="bold", pad=14,
    color="#222222",
)

# ── Source note ───────────────────────────────────────────────────────
ax.text(
    0.99, 0.01,
    "数据来源: 各模型官方 API 定价 & 公开基准测试 (2025)",
    transform=ax.transAxes, fontsize=7.5, color="#888888",
    ha="right", va="bottom", style="italic",
)

# ── Save ──────────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fig7-cost-accuracy.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
print(f"✅ Saved → {out_path}")
plt.close(fig)
