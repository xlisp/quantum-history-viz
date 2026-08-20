"""1926 · 薛定谔 & 玻恩 —— 波函数登场

薛定谔方程（含时 / 定态）：
    iħ ∂ψ/∂t = Ĥψ,        Ĥψ = Eψ,        Ĥ = −ħ²/2m ∇² + V(x)

玻恩（1926）给出概率诠释：|ψ(x)|² dx 是在 x 附近找到粒子的概率——
薛定谔本人一辈子不接受这个解释，但它是量子力学的地基。

本模块用 PyTorch 做三件事（全部取自然单位 ħ = m = 1）：
  1) 有限差分离散哈密顿量 → torch.linalg.eigh 求本征值，
     验证谐振子 E_n = (n+1/2)ħω 与无限深方势阱 E_n = n²π²ħ²/2mL²；
  2) split-step 傅里叶法（torch.fft，复数张量）演化高斯波包穿过势垒，
     数值透射率与解析公式逐点对照 —— 这就是量子隧穿；
  3) 双缝实验：一个一个电子打上去，屏幕上逐渐「长出」干涉条纹（玻恩规则）。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import matplotlib.pyplot as plt

from .style import use_style, finish, section, C, SERIES, SEQ_BLUE, INK, INK_2, INK_MUTED

DT = torch.float64
CT = torch.complex128


# ------------------------------------------------------------ 1. 定态本征值
def solve_tise(x: torch.Tensor, V: torch.Tensor, k: int = 8):
    """三点差分离散 Ĥ = −½ d²/dx² + V，返回前 k 个本征值与本征函数。"""
    n = x.numel()
    dx = float(x[1] - x[0])
    main = 1.0 / dx ** 2 + V
    off = -0.5 / dx ** 2 * torch.ones(n - 1, dtype=DT)
    Hm = torch.diag(main) + torch.diag(off, 1) + torch.diag(off, -1)
    ev, evec = torch.linalg.eigh(Hm)
    psi = evec[:, :k].T
    psi = psi / torch.sqrt((psi ** 2).sum(dim=1, keepdim=True) * dx)
    return ev[:k], psi


def fig_eigenstates():
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))

    # (a) 谐振子 V = x²/2  →  E_n = n + 1/2
    x = torch.linspace(-8, 8, 1400, dtype=DT)
    V = 0.5 * x ** 2
    E, psi = solve_tise(x, V, 6)
    ax = axes[0]
    ax.plot(x, V, color=INK_MUTED, lw=1.6, label="势 V(x) = x²/2")
    for n in range(6):
        ax.axhline(float(E[n]), color=INK_MUTED, lw=0.7, ls=":", xmin=0.05, xmax=0.95)
        ax.plot(x, psi[n] * 1.1 + float(E[n]), color=SERIES[n % 8], lw=1.8)
        ax.text(6.0, float(E[n]) + 0.12, f"n={n}  E={float(E[n]):.4f}",
                fontsize=8.5, color=SERIES[n % 8])
    ax.set_ylim(-0.4, 7.2); ax.set_xlim(-7, 7)
    ax.set_xlabel("x"); ax.set_ylabel("能量 E  /  ψ_n(x) 偏置在各自能级上")
    ax.set_title("谐振子：数值本征值 vs (n+½)ħω")
    ax.legend(loc="upper left")
    err = max(abs(float(E[n]) - (n + 0.5)) for n in range(6))
    print(f"  谐振子前 6 个能级最大误差 = {err:.2e}  (理论 E_n = n + 1/2)")

    # (b) 无限深方势阱 →  E_n = n²π²/(2L²)
    L = 1.0
    x2 = torch.linspace(0, L, 1200, dtype=DT)
    V2 = torch.zeros_like(x2)
    V2[0] = V2[-1] = 1e8
    E2, psi2 = solve_tise(x2, V2, 5)
    ax = axes[1]
    for n in range(5):
        exact = (n + 1) ** 2 * math.pi ** 2 / (2 * L ** 2)
        ax.axhline(float(E2[n]), color=INK_MUTED, lw=0.7, ls=":")
        ax.plot(x2, psi2[n] * 3.0 + float(E2[n]), color=SERIES[n % 8], lw=1.8)
        ax.text(1.005, float(E2[n]), f"n={n+1}: 数值 {float(E2[n]):.2f} / 解析 {exact:.2f}",
                fontsize=8.5, color=SERIES[n % 8], va="center")
    ax.set_xlim(0, 1.0); ax.set_ylim(-8, 140)
    ax.set_xlabel("x / L"); ax.set_ylabel("能量 E")
    ax.set_title("无限深方势阱：E_n = n²π²ħ²/2mL²")

    return finish(fig, "03_eigenstates.png",
                  "把 Ĥ 离散成矩阵后，「求解薛定谔方程」就是一次 torch.linalg.eigh —— 能级量子化是边界条件的必然结果")


# ------------------------------------------------------------ 2. 隧穿
def split_step(psi0, x, V, dt, steps, snap_every=None):
    """split-step 傅里叶：ψ ← e^{-iVdt/2} F⁻¹ e^{-ik²dt/2} F e^{-iVdt/2} ψ"""
    n = x.numel()
    dx = float(x[1] - x[0])
    k = 2 * math.pi * torch.fft.fftfreq(n, d=dx).to(DT)
    expV = torch.exp(-0.5j * V.to(CT) * dt)
    expK = torch.exp(-0.5j * (k ** 2).to(CT) * dt)
    psi = psi0.clone()
    snaps = []
    for s in range(steps):
        psi = expV * torch.fft.ifft(expK * torch.fft.fft(expV * psi))
        if snap_every and s % snap_every == 0:
            snaps.append(psi.clone())
    return psi, snaps


def gaussian_packet(x, x0, k0, sigma):
    psi = torch.exp(-(x - x0) ** 2 / (2 * sigma ** 2)).to(CT) * torch.exp(1j * k0 * x.to(CT))
    dx = float(x[1] - x[0])
    return psi / torch.sqrt((psi.abs() ** 2).sum() * dx)


def barrier_T_analytic(E, V0, a):
    """矩形势垒的解析透射率（ħ=m=1）。"""
    E = np.asarray(E, dtype=float)
    T = np.empty_like(E)
    lo, hi = E < V0, E >= V0
    kap = np.sqrt(2 * (V0 - E[lo]))
    T[lo] = 1 / (1 + V0 ** 2 * np.sinh(kap * a) ** 2 / (4 * E[lo] * (V0 - E[lo])))
    if hi.any():
        kk = np.sqrt(2 * (E[hi] - V0))
        T[hi] = 1 / (1 + V0 ** 2 * np.sin(kk * a) ** 2 / (4 * E[hi] * (E[hi] - V0)))
    return T


def fig_tunneling():
    N, Lx = 4096, 400.0
    x = torch.linspace(-Lx / 2, Lx / 2, N, dtype=DT)
    a, V0 = 1.0, 2.0
    V = torch.where((x.abs() < a / 2), torch.full_like(x, V0), torch.zeros_like(x))
    # 边界吸收层，防止波包绕回来
    absorber = 8.0 * (torch.clamp(x.abs() - 160.0, min=0.0) / 40.0) ** 2

    fig = plt.figure(figsize=(13.2, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.34, wspace=0.22)

    # (a) 一次演化的快照：E < V0，仍有一部分穿过去
    k0 = math.sqrt(2 * 1.0)          # E = k0²/2 = 1.0 < V0 = 2.0
    psi0 = gaussian_packet(x, -45.0, k0, 10.0)
    dt, steps = 0.01, 5000
    _, snaps = split_step(psi0, x, V, dt, steps, snap_every=1000)
    ax = fig.add_subplot(gs[0, :])
    ax.axvspan(-a / 2, a / 2, color=INK_MUTED, alpha=0.30, lw=0)
    ax.annotate(f"势垒 V₀={V0}, 宽 a={a}", xy=(0, 0.052), xytext=(14, 0.052),
                fontsize=9, color=INK_2, va="center",
                arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1))
    ramp = [SEQ_BLUE[i] for i in (3, 6, 8, 10, 12)]   # 时间有序 → 单色顺序色阶
    for i, sn in enumerate(snaps):
        t = i * 1000 * dt
        ax.plot(x, sn.abs() ** 2, color=ramp[i % len(ramp)], lw=1.8, label=f"t = {t:.0f}")
    ax.text(-72, 0.049, "入射能量 E = 1.0 < V₀ = 2.0\n透射概率 ≈ 26%", fontsize=9.5,
            color=INK, bbox=dict(boxstyle="round,pad=0.45", fc="#eef4fd",
                                 ec=C["blue"], lw=1))
    ax.annotate("入射波与反射波干涉\n形成的驻波纹路", xy=(-14, 0.030), xytext=(-38, 0.033),
                fontsize=8.5, color=INK_MUTED,
                arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.9))
    ax.annotate("穿过去的那一部分", xy=(20, 0.014), xytext=(30, 0.030),
                fontsize=8.5, color=C["orange"],
                arrowprops=dict(arrowstyle="->", color=C["orange"], lw=0.9))
    ax.set_xlim(-80, 60); ax.set_ylim(0, 0.062)
    ax.set_xlabel("x"); ax.set_ylabel("|ψ(x,t)|²")
    ax.set_title("量子隧穿：能量不够，波包照样漏过去一部分")
    ax.legend(ncol=5, loc="upper right")

    # (b) 透射率 vs 能量：数值 vs 解析
    # 逐个能量做一次完整演化：波包在空间上要足够宽（动量分布才够窄），
    # 演化时间取「透射部分刚好完全越过势垒」，此时反射部分还没跑到边界，
    # 避免 FFT 的周期性边界让它绕回来污染结果。
    energies = np.linspace(0.2, 4.0, 20)
    dx = float(x[1] - x[0])
    x0, travel = -70.0, 140.0
    T_num = []
    for E in energies:
        kk = math.sqrt(2 * E)                        # 群速度 v = k (ħ=m=1)
        p0 = gaussian_packet(x, x0, kk, 10.0)
        steps = int(travel / (kk * 0.01))
        psiT, _ = split_step(p0, x, V, 0.01, steps)
        T_num.append(float((psiT.abs() ** 2)[x > a / 2 + 4].sum() * dx))
    T_ana = barrier_T_analytic(energies, V0, a)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(energies, T_ana, color=C["blue"], lw=2, label="解析解 T(E)")
    ax.plot(energies, T_num, "o", ms=6, color=C["orange"], label="split-step 数值")
    ax.axvline(V0, color=INK_MUTED, ls="--", lw=1)
    ax.text(V0 + 0.06, 0.06, "E = V₀", fontsize=9, color=INK_MUTED)
    ax.set_xlabel("入射能量 E"); ax.set_ylabel("透射率 T")
    ax.set_title("透射率：数值积分 vs 解析公式")
    ax.legend(loc="upper left")
    print(f"  隧穿 T(E=1.0)：数值 {T_num[5]:.3f} / 解析 {T_ana[5]:.3f}")

    # (c) 势垒宽度的指数压制
    ax = fig.add_subplot(gs[1, 1])
    widths = np.linspace(0.2, 4.0, 200)
    for i, E in enumerate([0.5, 1.0, 1.5]):
        T = barrier_T_analytic(np.full_like(widths, E), V0, widths)
        ax.semilogy(widths, T, color=SERIES[i], lw=2, label=f"E = {E}")
        kap = math.sqrt(2 * (V0 - E))
        ax.text(3.1, T[-1] * 2.2, f"κ={kap:.2f}", color=SERIES[i], fontsize=8.5)
    ax.set_xlabel("势垒宽度 a"); ax.set_ylabel("透射率 T（对数轴）")
    ax.set_title("T ≈ e^(−2κa)：宽一点，就指数级地穿不过去")
    ax.legend()

    return finish(fig, "03_tunneling.png",
                  "隧穿是恒星里氢聚变、α 衰变、扫描隧道显微镜和闪存的共同物理基础；κ = √(2m(V₀−E))/ħ")


# ------------------------------------------------------------ 3. 双缝
def fig_double_slit(seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    y = torch.linspace(-25, 25, 1600, dtype=DT)          # 屏幕坐标 (mm)
    lam, d, slit_w, Lscreen = 5e-7, 1.0e-4, 2.2e-5, 1.5  # m
    th = torch.atan(y * 1e-3 / Lscreen)
    alpha = math.pi * slit_w / lam * torch.sin(th)
    beta = math.pi * d / lam * torch.sin(th)
    single = (torch.sin(alpha) / torch.where(alpha.abs() < 1e-12,
                                             torch.full_like(alpha, 1e-12), alpha)) ** 2
    both = single * torch.cos(beta) ** 2                  # 双缝 = 单缝包络 × 干涉因子
    both = both / both.max()

    fig = plt.figure(figsize=(13.0, 6.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.25], hspace=0.42, wspace=0.28)

    counts = [50, 500, 5000, 60000]
    p = (both / both.sum())
    for i, nshot in enumerate(counts):
        ax = fig.add_subplot(gs[0, i])
        idx = torch.multinomial(p, nshot, replacement=True, generator=g)
        hits = y[idx]
        jitter = torch.rand(nshot, generator=g, dtype=DT) * 2 - 1
        ax.scatter(hits, jitter, s=0.7, c=C["blue"], alpha=0.35, linewidths=0)
        ax.set_xlim(-25, 25); ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([]); ax.set_title(f"{nshot} 个电子", fontsize=10.5)
        ax.grid(False)
        ax.set_facecolor("#f4f6fa")
        if i == 0:
            ax.set_ylabel("屏幕")

    ax = fig.add_subplot(gs[1, :])
    ax.plot(y, both, color=C["blue"], lw=2, label="双缝：|ψ₁+ψ₂|²（有干涉）")
    ax.plot(y, single / single.max() * 0.5, color=C["orange"], lw=2, ls="--",
            label="单缝包络 |ψ₁|²+|ψ₂|²（经典粒子预期，无条纹）")
    ax.fill_between(y.numpy(), 0, both.numpy(), color=C["blue"], alpha=0.08)
    for m in range(-4, 5):
        ym = float(torch.asin(torch.tensor(m * lam / d)) * Lscreen * 1e3)
        if abs(ym) < 25:
            ax.axvline(ym, color=INK_MUTED, lw=0.7, ls=":")
    ax.set_xlim(-25, 25)
    ax.set_xlabel("屏幕位置 y (mm)"); ax.set_ylabel("相对强度 |ψ|²")
    ax.set_title("玻恩规则：单个电子随机落点，大量电子累积出干涉条纹")
    ax.legend(loc="upper right")
    return finish(fig, "03_double_slit.png",
                  "先叠加振幅再取模方（ψ₁+ψ₂ 后平方），不是先取模方再相加 —— 这一步之差就是量子与经典的分界。条纹间距 Δy = λL/d")


def main():
    section("1926 · 薛定谔方程 & 玻恩概率诠释")
    print(__doc__)
    fig_eigenstates()
    fig_tunneling()
    fig_double_slit()


if __name__ == "__main__":
    use_style()
    main()
