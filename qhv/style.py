"""统一的绘图风格：中文字体、配色（通过 CVD 校验的分类色板）、保存工具。

配色来自 dataviz 参考色板（light 模式），按固定顺序取用，不循环生成新色。
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# --- 分类色板（固定顺序，禁止循环复用） -----------------------------------
C = {
    "blue":    "#2a78d6",
    "orange":  "#eb6834",
    "aqua":    "#1baf7a",
    "yellow":  "#eda100",
    "magenta": "#e87ba4",
    "green":   "#008300",
    "violet":  "#4a3aa7",
    "red":     "#e34948",
}
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"],
          C["magenta"], C["green"], C["violet"], C["red"]]

# 单色顺序色阶（blue 100 -> 700），用于连续量（热图）
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a887f"
GRID = "#e3e2dd"


def _pick_cjk_font() -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "Heiti TC",
                 "STHeiti", "Songti SC", "Noto Sans CJK SC", "Source Han Sans SC",
                 "Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei"):
        if name in installed:
            return name
    return "DejaVu Sans"


CJK = _pick_cjk_font()


def use_style() -> None:
    plt.rcParams.update({
        "font.sans-serif": [CJK, "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "axes.prop_cycle": plt.cycler(color=SERIES),
    })
    for ax_side in ("top", "right"):
        plt.rcParams[f"axes.spines.{ax_side}"] = False


def finish(fig, name: str, note: str | None = None) -> Path:
    """加脚注、收紧布局并保存到 figures/。"""
    if note:
        fig.text(0.01, 0.005, note, color=INK_MUTED, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.025 if note else 0, 1, 1))
    path = FIG_DIR / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {path.relative_to(FIG_DIR.parent)}")
    return path


def section(title: str) -> None:
    line = "─" * 66
    print(f"\n{line}\n{title}\n{line}")
