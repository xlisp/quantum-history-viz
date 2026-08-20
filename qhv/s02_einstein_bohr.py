"""1905 爱因斯坦 · 1913 玻尔 · 1924 德布罗意 —— 量子假设长出骨架

爱因斯坦（1905，光电效应，1921 年诺奖）：光本身就是一颗颗光量子
    E_kin,max = hν − W        （W = 逸出功；截止电压 U_s = (h/e)ν − W/e）
    → U_s-ν 直线的斜率只与 h/e 有关，与金属无关。密立根 1916 年用它测出 h。

玻尔（1913）：电子只能待在角动量量子化的定态上 L = nħ，跃迁时才吸收/放出光子
    E_n = −13.606 eV / n²,     hν = E_n₂ − E_n₁   （里德伯公式）

德布罗意（1924）：物质也是波，λ = h/p。玻尔的量子化条件立刻有了解释：
    2πr = nλ  ——  轨道上必须容下整数个波长，否则驻波自我抵消。
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from .style import use_style, finish, section, C, SERIES, INK, INK_2, INK_MUTED

H_TRUE, E_CHARGE = 6.62607015e-34, 1.602176634e-19
C_LIGHT, RY_EV = 2.99792458e8, 13.605693122994


# ------------------------------------------------------------------ 光电效应
def fig_photoelectric(seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    metals = {"钠 Na (W=2.28 eV)": 2.28, "锌 Zn (W=4.33 eV)": 4.33, "铂 Pt (W=6.35 eV)": 6.35}
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    all_nu, all_U = [], []
    for i, (name, W) in enumerate(metals.items()):
        nu0 = W * E_CHARGE / H_TRUE
        nu = torch.linspace(nu0 * 1.02, nu0 * 1.9, 9, dtype=torch.float64)
        U = (H_TRUE * nu / E_CHARGE - W) + 0.06 * torch.randn(9, generator=g, dtype=torch.float64)
        ax.plot(nu * 1e-15, U, "o", ms=6, color=SERIES[i], label=name)
        line = torch.linspace(nu0, nu0 * 1.95, 2, dtype=torch.float64)
        ax.plot(line * 1e-15, H_TRUE * line / E_CHARGE - W, color=SERIES[i], lw=2, alpha=0.8)
        ax.plot([nu0 * 1e-15], [0], "v", ms=9, color=SERIES[i])
        ax.annotate(f"截止频率 ν₀={nu0*1e-15:.2f}", (nu0 * 1e-15, 0),
                    textcoords="offset points", xytext=(4, -18), color=SERIES[i], fontsize=8.5)
        all_nu.append(nu); all_U.append(U)

    # 用 PyTorch 梯度下降做一次「带共享斜率的多组线性回归」：
    # 三种金属截距不同（逸出功不同），但斜率必须是同一个 h/e。
    # 频率以 1e15 Hz 为单位，保证数值条件良好。
    nu_s = torch.cat(all_nu) * 1e-15
    U = torch.cat(all_U)
    idx = torch.cat([torch.full((9,), i) for i in range(len(metals))])
    p_slope = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)   # 单位 V/(1e15 Hz)
    offs = torch.zeros(len(metals), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([p_slope, offs], lr=0.05)
    for _ in range(4000):
        opt.zero_grad()
        loss = ((p_slope * nu_s + offs[idx] - U) ** 2).mean()
        loss.backward(); opt.step()
    h_fit = float(p_slope) * 1e-15 * E_CHARGE
    print(f"  由 U_s-ν 直线斜率反推 h = {h_fit:.4e} J·s（真值 {H_TRUE:.4e}，误差 {abs(h_fit/H_TRUE-1)*100:.2f}%）")

    ax.axhline(0, color=INK_MUTED, lw=1)
    ax.set_xlabel("入射光频率 ν (×10¹⁵ Hz)")
    ax.set_ylabel("遏止电压 U_s (V)")
    ax.set_title("光电效应：三种金属，同一个斜率 h/e")
    ax.text(0.42, 0.06, f"三条直线严格平行\n拟合斜率 → h = {h_fit:.4e} J·s",
            transform=ax.transAxes, fontsize=10, color=INK,
            bbox=dict(boxstyle="round,pad=0.5", fc="#eef4fd", ec=C["blue"], lw=1))
    ax.legend(loc="upper left")
    return finish(fig, "02_photoelectric.png",
                  "经典波动图像预言动能随光强增大、与频率无关；实验却是 E_k = hν − W —— 光是粒子")


# ------------------------------------------------------------------ 波长转 RGB
def wavelength_to_rgb(wl_nm: float) -> tuple[float, float, float]:
    """把 380–780 nm 的可见光波长近似转成 RGB，用于画真实颜色的谱线。"""
    w = wl_nm
    if w < 380 or w > 780:
        return (0.35, 0.35, 0.35)
    if w < 440:   r, gg, b = -(w - 440) / 60, 0.0, 1.0
    elif w < 490: r, gg, b = 0.0, (w - 440) / 50, 1.0
    elif w < 510: r, gg, b = 0.0, 1.0, -(w - 510) / 20
    elif w < 580: r, gg, b = (w - 510) / 70, 1.0, 0.0
    elif w < 645: r, gg, b = 1.0, -(w - 645) / 65, 0.0
    else:         r, gg, b = 1.0, 0.0, 0.0
    if w > 700:   f = 0.3 + 0.7 * (780 - w) / 80
    elif w < 420: f = 0.3 + 0.7 * (w - 380) / 40
    else:         f = 1.0
    return (r * f, gg * f, b * f)


# ------------------------------------------------------------------ 玻尔模型
def bohr_levels(nmax: int = 7) -> np.ndarray:
    n = np.arange(1, nmax + 1)
    return -RY_EV / n ** 2


def fig_bohr():
    E = bohr_levels(7)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.6),
                                  gridspec_kw={"width_ratios": [1.15, 1]})

    for i, e in enumerate(E, start=1):
        ax.hlines(e, 0.05, 0.95, color=INK_2, lw=1.6)
        ax.text(0.97, e, f"n={i}   {e:.2f} eV", va="center", fontsize=9, color=INK_2)
    ax.hlines(0, 0.05, 0.95, color=INK_MUTED, lw=1.2, ls="--")
    ax.text(0.97, 0.15, "n=∞  电离 (0 eV)", fontsize=9, color=INK_MUTED)

    series = [("莱曼系 (紫外)", 1, C["violet"], 0.18),
              ("巴尔末系 (可见)", 2, C["aqua"], 0.45),
              ("帕邢系 (红外)", 3, C["orange"], 0.72)]
    for label, nf, col, x in series:
        for ni in range(nf + 1, 7):
            ax.add_patch(FancyArrowPatch((x, E[ni - 1]), (x, E[nf - 1]),
                                         arrowstyle="-|>", mutation_scale=11,
                                         color=col, lw=1.4, alpha=0.85))
        ax.text(x, 1.2, label, ha="center", fontsize=9.5, color=col, fontweight="bold")
    ax.set_ylim(-14.6, 2.4)
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])
    ax.set_ylabel("能量 E (eV)")
    ax.set_title("玻尔氢原子能级  E_n = −13.606/n² eV")
    ax.grid(False)

    # 巴尔末系真实谱线
    ax2.set_facecolor("#101014")
    lines = []
    for ni in range(3, 10):
        # E_photon(eV) = 13.606·(1/2² − 1/n²)，λ(nm) = 1239.84 / E(eV)
        lam_nm = 1239.841984 / (RY_EV * (1 / 4 - 1 / ni ** 2))
        lines.append((ni, lam_nm))
    for ni, lam_nm in lines:
        col = wavelength_to_rgb(lam_nm)
        ax2.axvline(lam_nm, color=col, lw=2.6 if ni < 7 else 1.4)
        if ni <= 6:
            ax2.text(lam_nm, 0.92, f"H{'αβγδ'[ni-3]}\n{lam_nm:.1f} nm",
                     rotation=0, ha="center", va="top", fontsize=8.5, color="white")
    ax2.set_xlim(360, 700)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel("波长 λ (nm)")
    ax2.set_title("巴尔末系：肉眼可见的氢光谱")
    ax2.grid(False)
    print("  巴尔末系: " + ", ".join(f"n={ni}→2: {l:.1f} nm" for ni, l in lines[:4]))
    return finish(fig, "02_bohr_spectrum.png",
                  "里德伯公式 1/λ = R(1/n_f² − 1/n_i²) 从玻尔量子化条件直接推出；1885 年巴尔末凑出的经验公式终于有了物理来源")


# ------------------------------------------------------------------ 德布罗意驻波
def fig_de_broglie():
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.9), subplot_kw={"projection": "polar"})
    for ax, n in zip(axes, [3, 4, 5, 4.35]):
        th = np.linspace(0, 2 * np.pi, 2000)
        r = 1 + 0.16 * np.sin(n * th)
        ok = abs(n - round(n)) < 1e-9
        col = C["blue"] if ok else C["red"]
        ax.plot(th, np.ones_like(th), color=INK_MUTED, lw=1, ls="--")
        ax.plot(th, r, color=col, lw=2.2)
        if not ok:   # 绕两圈，看到波形对不上、自我抵消
            th2 = np.linspace(0, 4 * np.pi, 4000)
            ax.plot(th2 % (2 * np.pi), 1 + 0.16 * np.sin(n * th2), color=col, lw=1.0, alpha=0.4)
        ax.set_title(f"n = {n}" + ("  ✓ 驻波" if ok else "  ✗ 相位对不上"),
                     color=col, fontsize=11, pad=14)
        ax.set_ylim(0, 1.35)
        ax.set_yticks([]); ax.set_xticks([])
        ax.grid(False)
    fig.suptitle("德布罗意：2πr = nλ —— 只有整数个波长的轨道才能稳定存在",
                 fontsize=13, fontweight="bold", y=1.04)
    return finish(fig, "02_de_broglie_standing_wave.png",
                  "λ = h/p 代入 2πr = nλ 立刻给出玻尔的 L = mvr = nħ。1927 年戴维孙-革末电子衍射实验证实了物质波")


def main():
    section("1905–1924 · 爱因斯坦 / 玻尔 / 德布罗意 —— 光是粒子，物质是波")
    print(__doc__)
    fig_photoelectric()
    fig_bohr()
    fig_de_broglie()


if __name__ == "__main__":
    use_style()
    main()
