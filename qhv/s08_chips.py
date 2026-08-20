"""芯片里的量子力学 —— 一年几千亿美元的量子技术，早就在你口袋里

人们说起「量子技术」总想到量子计算机，但量子力学最成功、最赚钱的应用是半导体：
没有能带论就没有晶体管，没有隧穿效应就没有闪存、齐纳二极管和扫描隧道显微镜，
而隧穿同时也是摩尔定律撞上的那堵墙。

这一章把前面的抽象公式接到硅片上：

  1) 能带的「涌现」：单个硅原子只有分立能级，10²³ 个原子排成周期晶格后，
     能级并成能带，布里渊区边界上打开带隙 E_g —— 导体/半导体/绝缘体的全部区别
     就是这个数和电子填到哪里。用平面波展开 + torch.linalg.eigh 直接算出来。
  2) 载流子统计：费米-狄拉克分布 + 本征载流子浓度 n_i ∝ exp(−E_g/2k_BT)。
     这解释了为什么锗器件怕热、硅是甜点、而 SiC/GaN 能上高温高压。
  3) 隧穿在芯片里的两副面孔：
     · 有害：MOS 栅氧化层薄到 1 nm 出头时，电子直接隧穿过去 → 待机漏电爆炸，
       2007 年 Intel 45nm 被迫改用 HfO₂ 高 k 介质，物理厚度加厚而电容不变。
     · 有用：闪存靠 Fowler-Nordheim 隧穿写入/擦除；江崎二极管靠带间隧穿
       做出负微分电阻（江崎玲於奈 1957 年发现，1973 年诺奖）。
  4) 反向击穿的两种机制：
     · 齐纳击穿 = 带间量子隧穿（低压、高掺杂、温度系数为负）
     · 雪崩击穿 = 碰撞电离连锁（高压、低掺杂、温度系数为正）
     两者在 5.6 V 附近温度系数相消 —— 这就是经典的 5.6 V 电压基准二极管。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .style import use_style, finish, section, C, SERIES, SEQ_BLUE, INK, INK_2, INK_MUTED

DT = torch.float64

# ---- 物理常数（SI，除非另注） ----
Q = 1.602176634e-19          # C
HBAR = 1.054571817e-34       # J·s
M0 = 9.1093837015e-31        # kg
KB_EV = 8.617333262e-5       # eV/K
EPS0 = 8.8541878128e-12      # F/m

# ---- 材料参数（300 K） ----
MATERIALS = {
    "锗 Ge":   dict(Eg=0.661, Nc=1.04e19, Nv=6.0e18, color=C["orange"]),
    "硅 Si":   dict(Eg=1.12,  Nc=2.8e19,  Nv=1.04e19, color=C["blue"]),
    "砷化镓 GaAs": dict(Eg=1.424, Nc=4.7e17, Nv=9.0e18, color=C["aqua"]),
    "碳化硅 4H-SiC": dict(Eg=3.26, Nc=1.7e19, Nv=2.5e19, color=C["violet"]),
}


# ================================================================ 1. 能带
def bands_plane_wave(V1: float, nk: int = 241, M: int = 8):
    """一维周期势 V(x) = 2·V1·cos(2πx/a) 中的电子能带。

    平面波基 |k + nG〉（G = 2π/a），矩阵元
        H[n,m] = δ_nm·(k+nG)²  +  V1·δ_{|n−m|,1}
    取 a = 1、ħ²/2m = 1 的自然单位。对每个 k 做一次厄米对角化即得能带。
    """
    G = 2 * math.pi
    ns = torch.arange(-M, M + 1, dtype=DT)
    ks = torch.linspace(-math.pi, math.pi, nk, dtype=DT)
    off = V1 * (torch.abs(ns[:, None] - ns[None, :]) == 1).to(DT)
    bands = []
    for k in ks:
        Hm = torch.diag((k + ns * G) ** 2) + off
        bands.append(torch.linalg.eigvalsh(Hm))
    return ks.numpy(), torch.stack(bands).numpy()


def fig_band_structure():
    fig = plt.figure(figsize=(13.6, 8.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.92], hspace=0.42, wspace=0.26)

    for j, V1 in enumerate([0.0, 1.0, 4.0]):
        ks, bands = bands_plane_wave(V1)
        ax = fig.add_subplot(gs[0, j])
        for b in range(4):
            ax.plot(ks / math.pi, bands[:, b], color=SERIES[b], lw=2.2)
        gap = bands[:, 1].min() - bands[:, 0].max()
        ax.set_ylim(0, 60)
        ax.set_xlabel("波矢 k  (π/a 为单位)")
        if j == 0:
            ax.set_ylabel("能量 E  (ħ²/2ma² 为单位)")
        ax.set_title(f"V₁ = {V1:.0f}   带隙 = {max(gap,0):.2f}", fontsize=11)
        if gap > 0.05:
            ax.axhspan(bands[:, 0].max(), bands[:, 1].min(), color=C["red"], alpha=0.13, lw=0)
            ax.text(0, (bands[:, 0].max() + bands[:, 1].min()) / 2, "禁带",
                    ha="center", va="center", fontsize=9, color=C["red"], fontweight="bold")
        else:
            ax.text(0, 30, "自由电子抛物线\n（折叠到第一布里渊区）",
                    ha="center", fontsize=8.6, color=INK_2)
    fig.text(0.5, 0.965, "能带是「涌现」出来的：一个弱周期势就足以在布里渊区边界撕开禁带",
             ha="center", fontsize=13, fontweight="bold")

    # 带隙随势强度：小 V1 时严格等于 2|V1|（近自由电子微扰论）
    ax = fig.add_subplot(gs[1, 0])
    V1s = np.linspace(0, 5, 26)
    gaps = []
    for v in V1s:
        _, bands = bands_plane_wave(float(v), nk=81)
        gaps.append(max(bands[:, 1].min() - bands[:, 0].max(), 0.0))
    ax.plot(V1s, gaps, "o", ms=5, color=C["orange"], label="数值对角化")
    ax.plot(V1s, 2 * V1s, color=C["blue"], lw=2, label="微扰论 E_gap = 2|V₁|")
    ax.set_xlabel("周期势的傅里叶分量 V₁"); ax.set_ylabel("第一带隙")
    ax.set_title("① 带隙 = 两倍的势的傅里叶分量", fontsize=11)
    ax.legend(fontsize=8.5)
    err = max(abs(g - 2 * v) for g, v in zip(gaps[:8], V1s[:8]))
    print(f"  带隙数值 vs 微扰论 2|V₁|（小势区）最大偏差 = {err:.3f}")

    # 三类材料
    ax = fig.add_subplot(gs[1, 1:])
    kinds = [("金属\n（导体）", 0.0, 0.55, C["orange"]),
             ("半导体 Si\nE_g = 1.12 eV", 1.12, 0.0, C["blue"]),
             ("绝缘体 SiO₂\nE_g ≈ 9 eV", 9.0, 0.0, C["violet"])]
    for i, (name, eg, fill, col) in enumerate(kinds):
        x = i * 1.45
        scale = 0.42
        ax.add_patch(Rectangle((x, -3.2), 0.8, 3.2, fc=col, alpha=0.55, lw=0))       # 价带
        ax.text(x + 0.4, -1.6, "价带\n（填满）", ha="center", va="center",
                fontsize=8.2, color="white")
        top = eg * scale
        ax.add_patch(Rectangle((x, top), 0.8, 3.2, fc=col, alpha=0.16, lw=0))        # 导带
        if fill > 0:
            ax.add_patch(Rectangle((x, top), 0.8, 3.2 * fill, fc=col, alpha=0.55, lw=0))
            ax.text(x + 0.4, top + 0.9, "部分填充\n→ 自由导电", ha="center", va="center",
                    fontsize=8.2, color="white")
        else:
            ax.annotate("", xy=(x + 0.4, top), xytext=(x + 0.4, 0),
                        arrowprops=dict(arrowstyle="<->", color=C["red"], lw=1.6))
            ax.text(x + 0.46, top / 2, f"E_g\n{eg} eV", fontsize=8.4, color=C["red"],
                    va="center", ha="left", linespacing=1.4)
        ax.text(x + 0.4, -3.6, name, ha="center", va="top", fontsize=9.5, color=INK)
    ax.text(4.5, 1.0,
            "常温下电子靠热激发跨越带隙，\n"
            "概率 ∝ exp(−E_g/2k_BT)，k_BT ≈ 0.026 eV：\n\n"
            "  · Si（1.12 eV）→ 本征载流子 ~10¹⁰ cm⁻³\n"
            "    掺一点杂质就能把导电性调 10⁶ 倍 → 晶体管\n"
            "  · SiO₂（9 eV）→ 完全绝缘，做栅介质\n"
            "  · 金属 → 导带本身没填满，无需跨越",
            fontsize=9, color=INK_2, va="center", linespacing=1.6)
    ax.set_xlim(-0.3, 9.6); ax.set_ylim(-5.2, 5.4)
    ax.axis("off")
    ax.set_title("② 导体 / 半导体 / 绝缘体：区别只在带隙与填充", fontsize=11)
    return finish(fig, "08_band_structure.png",
                  "孤立原子的分立能级 → 10²³ 个原子的周期排列 → 能带 + 禁带。整个半导体工业就建立在「带隙」这两个字上")


# ================================================================ 2. 载流子
def n_intrinsic(T, Eg, Nc300, Nv300):
    """本征载流子浓度 n_i(T) = √(Nc·Nv)·exp(−Eg/2k_BT)，态密度按 T^{3/2} 标定。"""
    Nc = Nc300 * (T / 300.0) ** 1.5
    Nv = Nv300 * (T / 300.0) ** 1.5
    return np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * KB_EV * T))


def solve_fermi(T, Nd, Eg, Nc300, Nv300, Ed_depth=0.045):
    """给定温度与施主浓度，用二分法解电中性方程 n = p + N_D⁺，返回电子浓度 n。"""
    Nc = Nc300 * (T / 300.0) ** 1.5
    Nv = Nv300 * (T / 300.0) ** 1.5
    kT = KB_EV * T
    lo, hi = -0.5, Eg + 0.5                       # 以价带顶为 0，导带底为 Eg
    for _ in range(200):
        Ef = 0.5 * (lo + hi)
        n = Nc * math.exp(-(Eg - Ef) / kT)
        p = Nv * math.exp(-Ef / kT)
        Ndp = Nd / (1 + 2 * math.exp((Ef - (Eg - Ed_depth)) / kT))
        if n - p - Ndp > 0:
            hi = Ef
        else:
            lo = Ef
    return n, Ef


def fig_carriers():
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))

    # ① 费米-狄拉克分布
    ax = axes[0]
    E = np.linspace(-0.35, 0.35, 600)
    ramp = [SEQ_BLUE[i] for i in (12, 9, 6, 3)]
    for col, T in zip(ramp, [4, 100, 300, 900]):
        f = 1 / (1 + np.exp(E / (KB_EV * T)))
        ax.plot(E, f, color=col, lw=2.2, label=f"T = {T} K")
    ax.axvline(0, color=INK_MUTED, lw=1, ls=":")
    ax.text(0.012, 0.52, "E = E_F 处恒为 1/2", fontsize=8.8, color=INK_2)
    ax.set_xlabel("E − E_F (eV)"); ax.set_ylabel("占据概率 f(E)")
    ax.set_title("① 费米-狄拉克分布：泡利原理的统计后果")
    ax.legend(fontsize=8.5)

    # ② 本征载流子浓度
    ax = axes[1]
    T = np.linspace(200, 700, 300)
    for name, m in MATERIALS.items():
        ni = n_intrinsic(T, m["Eg"], m["Nc"], m["Nv"])
        ax.semilogy(1000 / T, ni, color=m["color"], lw=2.2,
                    label=f"{name}  E_g={m['Eg']} eV")
        ni300 = n_intrinsic(np.array([300.0]), m["Eg"], m["Nc"], m["Nv"])[0]
        print(f"  n_i(300K) {name}: {ni300:.2e} cm⁻³")
    ax.axhline(1e15, color=INK_MUTED, ls="--", lw=1.4)
    ax.text(2.7, 3e15, "典型掺杂浓度 10¹⁵ cm⁻³\n本征载流子一旦超过它，器件就失控",
            fontsize=8.6, color=INK_2)
    ax.set_xlabel("1000 / T  (K⁻¹)"); ax.set_ylabel("本征载流子浓度 n_i (cm⁻³)")
    ax.set_ylim(1e2, 1e20)
    ax.set_title("② 为什么锗被淘汰、SiC 能上高温")
    ax.legend(fontsize=7.8, loc="lower left")

    # ③ 掺杂硅的电子浓度：冻析 / 饱和 / 本征
    ax = axes[2]
    Ts = np.linspace(20, 800, 400)
    Nd = 1e15
    ns = np.array([solve_fermi(float(t), Nd, 1.12, 2.8e19, 1.04e19)[0] for t in Ts])
    ax.semilogy(1000 / Ts, ns, color=C["blue"], lw=2.4, label="n(T)，N_D = 10¹⁵ cm⁻³")
    ax.semilogy(1000 / Ts, n_intrinsic(Ts, 1.12, 2.8e19, 1.04e19), color=C["orange"],
                lw=2, ls="--", label="本征 n_i(T)")
    ax.axhline(Nd, color=INK_MUTED, lw=1, ls=":")
    for x0, x1, name, col in [(0.0, 1.9, "本征区\n器件失效", C["red"]),
                              (1.9, 12, "饱和区\n芯片正常工作", C["aqua"]),
                              (12, 50, "冻析区\n杂质冻住", C["violet"])]:
        ax.axvspan(x0, x1, color=col, alpha=0.08, lw=0)
        ax.text((min(x1, 26) + x0) / 2, 3e16, name, ha="center", fontsize=8.6,
                color=col, fontweight="bold")
    ax.set_xlim(0, 26); ax.set_ylim(1e9, 1e18)
    ax.set_xlabel("1000 / T  (K⁻¹)"); ax.set_ylabel("电子浓度 n (cm⁻³)")
    ax.set_title("③ 硅器件的温度窗口从哪来")
    ax.legend(fontsize=8.2, loc="lower right")
    n_400 = solve_fermi(400.0, Nd, 1.12, 2.8e19, 1.04e19)[0]
    print(f"  掺杂硅 N_D=1e15：n(400K) = {n_400:.3e} cm⁻³（仍被掺杂钉住）")
    return finish(fig, "08_carriers.png",
                  "电中性方程 n = p + N_D⁺ 用二分法逐点求解费米能级；三段式曲线是每本半导体教材的第一张图，也是「结温上限」的物理来源")


# ================================================================ 3. 隧穿在芯片里
def kappa(phi_eV: float, m_ratio: float) -> float:
    """势垒中的衰减常数 κ = √(2m*φ)/ħ，返回单位 nm⁻¹。"""
    return math.sqrt(2 * m_ratio * M0 * phi_eV * Q) / HBAR * 1e-9


def fig_tunneling_in_chips():
    fig = plt.figure(figsize=(14.0, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.40, wspace=0.24)

    # ① 栅氧化层直接隧穿
    ax = fig.add_subplot(gs[0, 0])
    k_ox = kappa(3.1, 0.5)          # Si/SiO₂ 势垒 3.1 eV，SiO₂ 中隧穿有效质量 ≈0.5 m₀
    k_hk = kappa(1.5, 0.2)          # Si/HfO₂ 势垒 ≈1.5 eV，m* ≈0.2 m₀
    print(f"  SiO₂ 衰减常数 κ = {k_ox:.2f} nm⁻¹  →  每薄 0.1 nm，漏电 ×{math.exp(2*k_ox*0.1):.1f}")
    print(f"  HfO₂ 衰减常数 κ = {k_hk:.2f} nm⁻¹")
    t = np.linspace(0.8, 7.0, 400)
    J0 = 1.0 / math.exp(-2 * k_ox * 1.5)        # 标定：t=1.5 nm 时 J ≈ 1 A/cm²
    J = J0 * np.exp(-2 * k_ox * t)
    ax.semilogy(t, J, color=C["blue"], lw=2.4, label="SiO₂ 直接隧穿 J ∝ e^(−2κt)")
    ax.semilogy(t, J0 * np.exp(-2 * k_hk * t), color=C["aqua"], lw=2.2, ls="--",
                label="HfO₂：势垒更矮，但可以做得更厚")
    nodes = [("180 nm 节点\nt=3.5 nm", 3.5), ("130 nm\n2.2 nm", 2.2),
             ("90 nm\n1.7 nm", 1.7), ("65 nm\n1.2 nm", 1.2)]
    for name, tt in nodes:
        jj = J0 * math.exp(-2 * k_ox * tt)
        ax.plot([tt], [jj], "o", ms=7, color=C["orange"], zorder=5)
        ax.text(tt - 0.13, jj * 5.0, name, ha="right", fontsize=8.0,
                color=C["orange"], linespacing=1.35)
    ax.axhline(1e2, color=C["red"], ls="--", lw=1.6)
    ax.text(7.1, 4e2, "漏电上限：待机功耗失控", ha="right", fontsize=8.8, color=C["red"])

    # 相同等效电容（EOT = 1 nm）下，HfO₂ 的物理厚度是 SiO₂ 的 ε_HfO₂/ε_SiO₂ = 25/3.9 倍
    eot = 1.0
    t_hk = eot * 25 / 3.9
    J_hk = J0 * math.exp(-2 * k_hk * t_hk)
    J_ox = J0 * math.exp(-2 * k_ox * eot)
    ax.plot([eot], [J_ox], "s", ms=9, color=C["red"], zorder=6)
    ax.plot([t_hk], [J_hk], "s", ms=9, color=C["aqua"], zorder=6)
    ax.annotate(f"EOT=1 nm 若用纯 SiO₂\nJ ≈ {J_ox:.0e} A/cm²，不可用",
                xy=(eot, J_ox), xytext=(1.9, 3e8), fontsize=8.4, color=C["red"],
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=1))
    ax.annotate(f"同样 EOT 改用 HfO₂：物理厚度 {t_hk:.1f} nm\nJ ≈ {J_hk:.0e} A/cm²（理想估计）",
                xy=(t_hk, J_hk), xytext=(4.15, 4e-6), fontsize=8.4, color=C["aqua"],
                arrowprops=dict(arrowstyle="->", color=C["aqua"], lw=1))
    ax.set_xlim(0.7, 7.3)
    ax.set_ylim(1e-14, 1e10)
    ax.set_xlabel("栅介质物理厚度 t (nm)")
    ax.set_ylabel("栅极漏电流密度 J (A/cm²)")
    ax.set_title("① 摩尔定律撞上的量子墙：栅氧再薄就漏电")
    ax.legend(loc="upper right", fontsize=8.2)

    # ② Fowler-Nordheim 隧穿（闪存的写入/擦除）
    ax = fig.add_subplot(gs[0, 1])
    phi, mstar = 3.1, 0.5
    A = Q ** 3 / (8 * math.pi * 6.62607015e-34 * phi * Q * mstar)      # 简化系数
    B = 8 * math.pi * math.sqrt(2 * mstar * M0) * (phi * Q) ** 1.5 / (3 * Q * 6.62607015e-34)
    Efield = np.linspace(4e6, 14e6, 300) * 100        # V/m
    Jfn = A * Efield ** 2 * np.exp(-B / Efield)
    ax.semilogy(Efield / 1e8, Jfn, color=C["blue"], lw=2.4)
    ax.axvspan(8, 12, color=C["aqua"], alpha=0.12, lw=0)
    ax.text(10, Jfn.max() * 1e-3, "闪存编程/擦除\n工作区 8–12 MV/cm",
            ha="center", fontsize=9, color=C["aqua"], fontweight="bold")
    ax.set_xlabel("氧化层电场 E (MV/cm)")
    ax.set_ylabel("F-N 隧穿电流密度 J (A/cm²)")
    ax.set_title("② 闪存靠隧穿写入：Fowler-Nordheim 电流")
    ins = ax.inset_axes([0.60, 0.14, 0.36, 0.34])
    ins.plot(1 / (Efield / 1e8), np.log(Jfn / Efield ** 2), color=C["orange"], lw=1.8)
    ins.set_title("F-N 图：ln(J/E²)~1/E 是直线", fontsize=7.5, pad=3)
    ins.tick_params(labelsize=6.5)
    ins.set_xlabel("1/E", fontsize=7); ins.set_ylabel("ln(J/E²)", fontsize=7)
    print(f"  F-N 参数 B = {B/1e8:.1f} MV/cm；10 MV/cm 处 J = "
          f"{A*(1e9)**2*math.exp(-B/1e9):.2e} A/cm²")

    # ③ 江崎二极管：负微分电阻
    ax = fig.add_subplot(gs[1, 0])
    V = np.linspace(0, 0.6, 500)
    Vp, Ip, Vv = 0.06, 10.0, 0.35
    I_tunnel = Ip * (V / Vp) * np.exp(1 - V / Vp)
    I_excess = 1.2 * np.exp((V - Vv) / 0.10)
    I_diode = 0.06 * (np.exp(V / 0.052) - 1)
    I = I_tunnel + I_excess + I_diode
    ax.plot(V * 1e3, I, color=C["blue"], lw=2.4)
    ndr = (np.gradient(I, V) < 0)
    ax.fill_between(V * 1e3, 0, I, where=ndr, color=C["red"], alpha=0.15)
    ax.plot([Vp * 1e3], [I[np.argmin(abs(V - Vp))]], "o", ms=8, color=C["orange"])
    ax.annotate("峰值 I_p：带间隧穿", xy=(Vp * 1e3, Ip), xytext=(108, 11.7),
                fontsize=9, color=C["orange"],
                arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1))
    ax.text(205, 2.6, "负微分电阻区\ndI/dV < 0\n→ 可做振荡器/放大器",
            fontsize=9, color=C["red"], fontweight="bold")
    ax.set_xlabel("正向电压 V (mV)"); ax.set_ylabel("电流 I (mA，示意标度)")
    ax.set_xlim(0, 460); ax.set_ylim(0, 13)
    ax.set_title("③ 江崎二极管（1957）：隧穿造出的负电阻")

    # ④ 隧穿在芯片里的位置一览
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title("④ 一颗芯片里，隧穿出现在哪些地方", fontsize=11.5)
    rows = [
        ("闪存 / SSD", "F-N 隧穿把电子打进浮栅", "有用", C["aqua"]),
        ("齐纳二极管", "带间隧穿造出稳定的基准电压", "有用", C["aqua"]),
        ("磁隧道结 MRAM", "自旋相关隧穿 → 读出磁化方向", "有用", C["aqua"]),
        ("扫描隧道显微镜", "隧穿电流对间距指数敏感 → 看见原子", "有用", C["aqua"]),
        ("MOS 栅极漏电", "栅氧 <1.5 nm 后直接隧穿，待机功耗爆炸", "有害", C["red"]),
        ("短沟道源漏穿通", "沟道 <5 nm 时电子直接从源穿到漏", "有害", C["red"]),
        ("氧化层老化 TDDB", "隧穿注入的电子逐渐打出缺陷 → 击穿失效", "有害", C["red"]),
    ]
    for i, (name, desc, tag, col) in enumerate(rows):
        y = 0.93 - i * 0.132
        ax.add_patch(Rectangle((0.005, y - 0.052), 0.055, 0.088, fc=col, alpha=0.75,
                               transform=ax.transAxes, lw=0, clip_on=False))
        ax.text(0.0325, y - 0.008, tag, transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
        ax.text(0.085, y - 0.008, name, transform=ax.transAxes, fontsize=9.8,
                color=INK, fontweight="bold", va="center")
        ax.text(0.34, y - 0.008, desc, transform=ax.transAxes, fontsize=9,
                color=INK_2, va="center")
    return finish(fig, "08_tunneling_in_chips.png",
                  "同一个 exp(−2κt) 既是闪存能存住数据 10 年的原因，也是栅氧不能再薄的原因——尺度差不到 1 nm，结果差 10 个数量级")


# ================================================================ 4. 击穿
def v_breakdown(Nb_cm3, Eg=1.12):
    """Sze 的经验公式：V_B ≈ 60·(E_g/1.1)^{3/2}·(N_B/10¹⁶)^{−3/4}  (V)"""
    return 60 * (Eg / 1.1) ** 1.5 * (np.asarray(Nb_cm3) / 1e16) ** -0.75


def fig_breakdown():
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.2))

    # ① 二极管 I-V 全景
    ax = axes[0, 0]
    Vt, Is = 0.02585, 1e-12
    curves = [(2.1e17, C["orange"], "重掺杂 → 齐纳击穿（隧穿）"),
              (1.0e16, C["blue"], "轻掺杂 → 雪崩击穿（碰撞电离）")]
    for Nb, col, name in curves:
        Vbr = float(v_breakdown(Nb))
        V = np.linspace(-75, 0.85, 6000)
        I_fwd = np.where(V > 0, Is * (np.exp(np.clip(V, 0, 0.85) / Vt) - 1), 0.0)
        I_rev = -Is - 1e-3 * np.exp((-V - Vbr) / (0.025 * Vbr)) * (V < 0)
        I = np.where(V > 0, I_fwd, I_rev)
        ax.plot(V, np.clip(I, -0.03, 0.03) * 1e3, color=col, lw=2.4, label=name)
        ax.annotate(f"V_BR ≈ {Vbr:.1f} V", xy=(-Vbr, -18), xytext=(-Vbr - 1.5, -27.5),
                    ha="center", fontsize=9, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1))
    ax.axhline(0, color=INK_MUTED, lw=0.9); ax.axvline(0, color=INK_MUTED, lw=0.9)
    ax.set_xlim(-72, 2.5); ax.set_ylim(-32, 32)
    ax.set_xlabel("端电压 V (V)"); ax.set_ylabel("电流 I (mA)")
    ax.set_title("① 掺杂浓度决定击穿电压与击穿机制")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.text(-70, 8, "反向：几乎不导通，\n直到击穿点电流垂直上冲",
            fontsize=8.6, color=INK_2, linespacing=1.5)
    ins = ax.inset_axes([0.52, 0.56, 0.33, 0.33])
    Vf = np.linspace(0, 0.8, 400)
    ins.plot(Vf, Is * (np.exp(Vf / Vt) - 1) * 1e3, color=INK_2, lw=1.8)
    ins.set_ylim(0, 30); ins.set_xlim(0, 0.8)
    ins.set_title("正向区放大：I ∝ e^(qV/kT)", fontsize=7.5, pad=3)
    ins.tick_params(labelsize=6.5)
    ins.set_xlabel("V (V)", fontsize=7); ins.set_ylabel("I (mA)", fontsize=7)

    # ② 击穿电压 vs 掺杂
    ax = axes[0, 1]
    Nb = np.logspace(14, 19, 300)
    Vb = v_breakdown(Nb)
    ax.loglog(Nb, Vb, color=C["blue"], lw=2.4, label="V_BR (Sze 经验式)")
    ax.axhspan(0.1, 4 * 1.12, color=C["orange"], alpha=0.12, lw=0)
    ax.axhspan(4 * 1.12, 6 * 1.12, color=C["yellow"], alpha=0.14, lw=0)
    ax.axhspan(6 * 1.12, 1e4, color=C["aqua"], alpha=0.10, lw=0)
    ax.text(1.6e14, 2.4, "V_BR < 4E_g/q ≈ 4.5 V：纯齐纳（隧穿）",
            fontsize=8.6, color=C["orange"])
    ax.text(1.6e14, 5.2, "4–6 E_g/q：两种机制并存", fontsize=8.6, color="#9a6a00")
    ax.text(1.6e14, 30, "V_BR > 6E_g/q ≈ 6.7 V：雪崩主导", fontsize=8.6, color=C["aqua"])
    for n, lab in [(1e15, "功率器件\n轻掺杂"), (1e18, "齐纳二极管\n重掺杂")]:
        ax.plot([n], [float(v_breakdown(n))], "o", ms=8, color=C["red"], zorder=5)
        ax.text(n * 1.35, float(v_breakdown(n)), lab, fontsize=8.4, color=C["red"], va="center")
    ax.set_xlabel("轻掺杂一侧的掺杂浓度 N_B (cm⁻³)"); ax.set_ylabel("击穿电压 V_BR (V)")
    ax.set_ylim(1, 3e3)
    ax.set_title("② 想要高耐压，就得掺得淡、做得厚")
    ax.legend(fontsize=8.5)
    print(f"  V_BR：N=1e15 → {float(v_breakdown(1e15)):.0f} V；"
          f"N=1e18 → {float(v_breakdown(1e18)):.1f} V")

    # ③ 两种机制的场依赖
    ax = axes[1, 0]
    E = np.linspace(2e5, 1.2e6, 400)                    # V/cm
    B_kane = 4 * math.sqrt(2 * 0.2 * M0) * (1.12 * Q) ** 1.5 / (3 * Q * HBAR) / 100  # V/cm
    P_tunnel = np.exp(-B_kane / E)
    alpha_ii = 7.03e5 * np.exp(-1.231e6 / E)            # cm⁻¹，硅的碰撞电离系数
    ax.semilogy(E / 1e5, P_tunnel / P_tunnel.max(), color=C["orange"], lw=2.4,
                label=f"带间隧穿几率 ∝ exp(−B/E)，B = {B_kane/1e6:.1f} MV/cm")
    ax.semilogy(E / 1e5, alpha_ii / alpha_ii.max(), color=C["aqua"], lw=2.4,
                label="碰撞电离系数 α ∝ exp(−b/E)，b = 12.3 MV/cm")
    ax.set_xlabel("耗尽区峰值电场 E (10⁵ V/cm)")
    ax.set_ylabel("归一化几率（对数轴）")
    ax.set_ylim(1e-8, 3)
    ax.set_title("③ 两种击穿机制的电场门槛不同")
    ax.text(3.4, 3e-5,
            "隧穿需要极高电场，但势垒也薄\n→ 只在重掺杂的窄耗尽区里达到；\n\n"
            "雪崩需要电子在长耗尽区中加速够远\n→ 轻掺杂厚器件的宿命。",
            fontsize=8.6, color=INK_2, linespacing=1.6)
    ax.legend(loc="lower right", fontsize=8.2)

    # ④ 温度系数：齐纳负、雪崩正
    ax = axes[1, 1]
    Vz = np.linspace(2, 12, 200)
    tc = -0.11 + 0.0245 * (Vz - 2) ** 1.18            # 经验拟合，%/°C
    ax.plot(Vz, tc, color=C["blue"], lw=2.4)
    ax.axhline(0, color=INK_MUTED, lw=1.2, ls="--")
    zero = Vz[np.argmin(abs(tc))]
    ax.plot([zero], [0], "o", ms=10, color=C["red"], zorder=5,
            markeredgecolor="white", markeredgewidth=1.6)
    ax.annotate(f"≈ {zero:.1f} V：温度系数过零\n两种机制正好抵消",
                xy=(zero, 0), xytext=(zero + 0.7, -0.06), fontsize=9.2, color=C["red"],
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=1))
    ax.fill_between(Vz, tc, 0, where=(tc < 0), color=C["orange"], alpha=0.14)
    ax.fill_between(Vz, tc, 0, where=(tc > 0), color=C["aqua"], alpha=0.14)
    ax.text(2.4, -0.075, "齐纳（隧穿）主导\n温度↑ → 带隙↓ → 更易隧穿\n→ V_BR 下降（负温度系数）",
            fontsize=8.6, color=C["orange"], linespacing=1.5)
    ax.text(8.6, 0.10, "雪崩主导\n温度↑ → 声子散射↑ → 电子加速不起来\n→ V_BR 上升（正温度系数）",
            fontsize=8.6, color=C["aqua"], ha="center", linespacing=1.5)
    ax.set_xlabel("稳压二极管的标称电压 V_Z (V)")
    ax.set_ylabel("温度系数 (%/°C)")
    ax.set_title("④ 5.6 V 稳压管为什么是经典电压基准")
    print(f"  温度系数过零点 ≈ {zero:.2f} V（实测典型值 5.6 V）")
    return finish(fig, "08_breakdown.png",
                  "「击穿」不是一种现象而是两种：齐纳击穿是量子隧穿（可逆、可控），雪崩击穿是经典的碰撞电离连锁；两者都不等于「烧毁」——真正烧毁的是随后的热失控")


def main():
    section("芯片里的量子力学 —— 能带、隧穿、击穿")
    print(__doc__)
    fig_band_structure()
    fig_carriers()
    fig_tunneling_in_chips()
    fig_breakdown()


if __name__ == "__main__":
    use_style()
    main()
