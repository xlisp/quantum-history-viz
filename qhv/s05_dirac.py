"""1928 · 狄拉克 —— 把相对论和量子力学缝在一起，然后掉出一个反世界

薛定谔方程对时间一阶、对空间二阶，不满足洛伦兹协变。狄拉克要一个对时空都一阶的方程：

    (iħ γ^μ ∂_μ − mc) ψ = 0        等价地   iħ ∂ψ/∂t = (c α·p̂ + β mc²) ψ

代价是 ψ 必须是 4 分量旋量，系数 γ^μ 必须是矩阵，且满足克利福德代数

    {γ^μ, γ^ν} = γ^μγ^ν + γ^νγ^μ = 2 η^μν I₄

这个方程一次性白送了三样东西：
  · 电子自旋 1/2 与朗德因子 g = 2（不用手工塞进去）；
  · 精细结构（氢原子能级的相对论修正）；
  · 负能解 E = −√(p²c²+m²c⁴) —— 狄拉克 1931 年断言那是「反电子」，
    1932 年安德森在云室里找到了正电子。狄拉克说：「我的方程比我聪明。」

本模块：
  1) 数值验证克利福德代数与色散关系 E = ±√(p²c²+m²c⁴)（质量间隙 2mc²）；
  2) 严格演化 1+1 维狄拉克方程，看到「颤动」Zitterbewegung：
     正能解与负能解的干涉让 〈x〉以 ω_Z = 2mc²/ħ 抖动，幅度约为康普顿波长 ħ/mc；
  3) g−2：狄拉克给出 g=2，施温格 1948 年的 α/2π 修正与今天的实验对到小数点后 12 位。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import matplotlib.pyplot as plt

from .style import use_style, finish, section, C, SERIES, INK, INK_2, INK_MUTED

DT, CT = torch.float64, torch.complex128
I2 = torch.eye(2, dtype=CT)
SX = torch.tensor([[0, 1], [1, 0]], dtype=CT)
SY = torch.tensor([[0, -1j], [1j, 0]], dtype=CT)
SZ = torch.tensor([[1, 0], [0, -1]], dtype=CT)
ZERO = torch.zeros(2, 2, dtype=CT)


def dirac_gammas():
    """狄拉克表象下的 γ^0, γ^1, γ^2, γ^3。"""
    def blk(a, b, c, d):
        return torch.cat([torch.cat([a, b], 1), torch.cat([c, d], 1)], 0)
    g0 = blk(I2, ZERO, ZERO, -I2)
    gs = [blk(ZERO, s, -s, ZERO) for s in (SX, SY, SZ)]
    return [g0] + gs


def check_clifford() -> float:
    g = dirac_gammas()
    eta = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=DT))
    worst = 0.0
    for mu in range(4):
        for nu in range(4):
            anti = g[mu] @ g[nu] + g[nu] @ g[mu]
            target = 2 * eta[mu, nu] * torch.eye(4, dtype=CT)
            worst = max(worst, float((anti - target).abs().max()))
    return worst


# ------------------------------------------------------------------ 色散关系
def fig_dispersion():
    err = check_clifford()
    print(f"  克利福德代数 {{γ^μ,γ^ν}} = 2η^μν 最大偏差 = {err:.1e}")

    m, c = 1.0, 1.0
    p = np.linspace(-3, 3, 600)
    Epos = np.sqrt(p ** 2 * c ** 2 + m ** 2 * c ** 4)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0),
                                  gridspec_kw={"width_ratios": [1.1, 1]})
    ax.plot(p, Epos, color=C["blue"], lw=2.4, label="正能解 E = +√(p²c²+m²c⁴)  → 电子")
    ax.plot(p, -Epos, color=C["orange"], lw=2.4, label="负能解 E = −√(…)  → 正电子")
    ax.plot(p, p ** 2 / (2 * m) + m, color=INK_MUTED, lw=1.6, ls="--",
            label="非相对论近似 E ≈ mc² + p²/2m")
    ax.fill_between(p, -1, 1, color=INK_MUTED, alpha=0.10)
    ax.annotate("", xy=(0, 1), xytext=(0, -1),
                arrowprops=dict(arrowstyle="<->", color=C["red"], lw=1.8))
    ax.text(0.12, 0, "质量间隙\n2mc² = 1.022 MeV", color=C["red"], fontsize=9.5, va="center")
    ax.text(-2.9, -2.4, "狄拉克海：负能态全被填满，\n打出一个空穴 = 一个正电子",
            fontsize=9, color=C["orange"])
    ax.set_xlabel("动量 p (mc 为单位)"); ax.set_ylabel("能量 E (mc² 为单位)")
    ax.set_title("狄拉克色散关系：能谱一分为二")
    ax.legend(loc="upper center", fontsize=8.8)
    ax.set_ylim(-3.4, 3.4)

    # 右图：γ 矩阵的结构
    g = dirac_gammas()
    big = torch.cat([torch.cat([g[0].real, g[1].real], 1),
                     torch.cat([g[2].imag, g[3].real], 1)], 0).numpy()
    im = ax2.imshow(big, cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.axhline(3.5, color=INK, lw=1.5); ax2.axvline(3.5, color=INK, lw=1.5)
    for (i, j), lab in [((0, 0), "γ⁰"), ((0, 1), "γ¹"), ((1, 0), "Im γ²"), ((1, 1), "γ³")]:
        ax2.text(j * 4 + 1.5, i * 4 - 0.7, lab, ha="center", fontsize=11,
                 color=INK, fontweight="bold")
    for i in range(8):
        for j in range(8):
            if abs(big[i, j]) > 1e-9:
                ax2.text(j, i, f"{big[i,j]:+.0f}", ha="center", va="center",
                         fontsize=8, color="white" if abs(big[i, j]) > 0.5 else INK)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.grid(False)
    ax2.set_title("四个 4×4 的 γ 矩阵（狄拉克表象）")
    ax2.text(0.5, -0.08, f"数值验证 {{γ^μ,γ^ν}} = 2η^μν·I₄，最大偏差 {err:.0e}",
             transform=ax2.transAxes, ha="center", fontsize=9, color=INK_2)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    return finish(fig, "05_dirac_dispersion.png",
                  "1928 年狄拉克为了「开平方」而引入矩阵系数，负能解起初被认为是灾难，四年后变成了反物质的预言")


# ------------------------------------------------------------------ 颤动
def dirac_evolve_1d(psi0_k, k, m, c, t):
    """1+1 维狄拉克方程在动量空间的严格演化：
       H(k) = c k σx + m c² σz,  U = cos(Et)I − i sin(Et)/E · H(k)"""
    E = torch.sqrt((c * k) ** 2 + (m * c ** 2) ** 2)
    cos, sin = torch.cos(E * t), torch.sin(E * t) / E
    # 直接对每个 k 构造 2x2 演化矩阵作用在 (2, Nk) 的旋量上
    h11 = (m * c ** 2) * torch.ones_like(k)
    h12 = c * k
    u11 = cos.to(CT) - 1j * sin.to(CT) * h11.to(CT)
    u22 = cos.to(CT) + 1j * sin.to(CT) * h11.to(CT)
    u12 = -1j * sin.to(CT) * h12.to(CT)
    out = torch.stack([u11 * psi0_k[0] + u12 * psi0_k[1],
                       u12 * psi0_k[0] + u22 * psi0_k[1]])
    return out


def fig_zitterbewegung():
    m = c = 1.0                                  # ħ = c = m = 1，长度单位 = 康普顿波长
    N, L = 4096, 200.0
    x = torch.linspace(-L / 2, L / 2, N, dtype=DT)
    dx = float(x[1] - x[0])
    k = 2 * math.pi * torch.fft.fftfreq(N, d=dx).to(DT)

    k0, sigma = 0.4, 2.0
    g = torch.exp(-x ** 2 / (2 * sigma ** 2)).to(CT) * torch.exp(1j * k0 * x.to(CT))
    g = g / torch.sqrt((g.abs() ** 2).sum() * dx)
    psi0 = torch.stack([g, torch.zeros_like(g)])           # 上旋量 = 正负能的叠加
    psi0_k = torch.stack([torch.fft.fft(psi0[0]), torch.fft.fft(psi0[1])])

    ts = torch.linspace(0, 30, 900, dtype=DT)
    xmean, dens = [], []
    for i, t in enumerate(ts):
        pk = dirac_evolve_1d(psi0_k, k, m, c, t)
        px = torch.stack([torch.fft.ifft(pk[0]), torch.fft.ifft(pk[1])])
        rho = (px.abs() ** 2).sum(0)
        rho = rho / (rho.sum() * dx)
        xmean.append(float((x * rho).sum() * dx))
        if i % 90 == 0:
            dens.append((float(t), rho.clone()))
    xmean = np.array(xmean)
    ts_np = ts.numpy()

    # 分解：把「平均漂移」和「颤动」分开
    drift = np.polyval(np.polyfit(ts_np, xmean, 1), ts_np)
    trembling = xmean - drift
    amp = trembling.max() - trembling.min()
    # 主频（去掉直流）
    fft = np.abs(np.fft.rfft(trembling - trembling.mean()))
    freqs = 2 * np.pi * np.fft.rfftfreq(len(ts_np), d=float(ts_np[1] - ts_np[0]))
    w_peak = freqs[np.argmax(fft)]
    w_theory = 2 * math.sqrt(k0 ** 2 + (m * c) ** 2)      # ω_Z = 2E(k)/ħ ≥ 2mc²/ħ
    print(f"  颤动频率 ω_Z(数值) = {w_peak:.3f}，理论 2E(k₀)/ħ = {w_theory:.3f}"
          f"（下限 2mc²/ħ = {2*m*c**2:.1f}，FFT 频率分辨率 ±{2*math.pi/30:.2f}）")
    print(f"  颤动幅度 ≈ {amp:.3f} ħ/mc ≈ {amp*3.86e-13:.1e} m")

    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.36, wspace=0.22)

    ax = fig.add_subplot(gs[0, :])
    for i, (t, rho) in enumerate(dens):
        ax.plot(x, rho, color=SERIES[i % 8], lw=1.6, label=f"t = {t:.0f}")
    ax.set_xlim(-15, 30); ax.set_xlabel("x (康普顿波长 ħ/mc)"); ax.set_ylabel("ρ(x,t)")
    ax.set_title("1+1 维狄拉克波包的严格演化")
    ax.legend(ncol=5, loc="upper right", fontsize=8.5)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ts_np, xmean, color=C["blue"], lw=1.8, label="〈x〉(t)")
    ax.plot(ts_np, drift, color=INK_MUTED, lw=1.6, ls="--", label="经典匀速漂移")
    ax.set_xlabel("t"); ax.set_ylabel("〈x〉")
    ax.set_title("① 位置期望值：直线上叠着抖动")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ts_np, trembling, color=C["orange"], lw=1.8)
    per = 2 * math.pi / (2 * m * c ** 2)
    for j in range(int(30 / per) + 1):
        ax.axvline(j * per, color=INK_MUTED, lw=0.6, ls=":")
    ax.set_xlabel("t"); ax.set_ylabel("〈x〉 − 漂移")
    ax.set_title("② 扣掉漂移后：Zitterbewegung 颤动")
    ax.text(0.02, 0.04,
            f"数值主频 ω = {w_peak:.2f}，理论 ω_Z = 2E(k₀)/ħ = {w_theory:.2f}\n"
            f"下限 2mc²/ħ = {2*m*c**2:.1f}；虚线为周期 πħ/mc² = {per:.2f}\n"
            f"幅度 ≈ {amp:.2f} ħ/mc ≈ {amp*3.86e-13:.0e} m",
            transform=ax.transAxes, fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc="#fdf1ea", ec=C["orange"], lw=1))
    return finish(fig, "05_zitterbewegung.png",
                  "颤动来自正能解与负能解的干涉项 e^{-i(E−(−E))t/ħ}；电子从来不是点，它在康普顿尺度上「抖」——1930 年薛定谔发现，2010 年被离子阱模拟实验直接看到")


# ------------------------------------------------------------------ g−2
def fig_g_minus_2():
    alpha = 7.2973525693e-3
    a_dirac = 0.0
    a_schwinger = alpha / (2 * math.pi)
    a_qed = 0.00115965218161          # QED 五圈理论值
    a_exp = 0.00115965218059          # 实验值 (Fan et al. 2023)
    print(f"  施温格一圈修正 α/2π = {a_schwinger:.11f}")
    print(f"  QED 理论 {a_qed:.14f} vs 实验 {a_exp:.14f}，相对差 {abs(a_qed/a_exp-1):.1e}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6),
                                  gridspec_kw={"width_ratios": [1, 1.05]})
    items = [("狄拉克 1928\ng = 2 精确", 2.0, C["blue"]),
             ("施温格 1948\ng = 2(1+α/2π)", 2 * (1 + a_schwinger), C["orange"]),
             ("QED 五圈理论", 2 * (1 + a_qed), C["aqua"]),
             ("实验 2023", 2 * (1 + a_exp), C["red"])]
    ys = np.arange(len(items))[::-1]
    for (label, g, col), y in zip(items, ys):
        ax.barh(y, g - 2, color=col, height=0.55)
        ax.text(max(g - 2, 0) + 4e-5, y, f"g−2 = {g-2:.9f}", va="center",
                fontsize=9, color=INK_2)
    ax.set_yticks(ys); ax.set_yticklabels([i[0] for i in items], fontsize=9)
    ax.set_xlim(0, 0.0038)
    ax.set_xlabel("反常磁矩 g − 2")
    ax.set_title("电子 g 因子：狄拉克给出 2，量子场论补上剩下的")
    ax.grid(True, axis="x")

    # 右：理论 vs 实验的逐位比较
    ax2.axis("off")
    txt = ("            理论(QED)   1.001 159 652 181 61\n"
           "            实验(2023)   1.001 159 652 180 59\n"
           "                         ─────────────────────\n"
           "            一致到       小数点后第 12 位")
    ax2.text(0.02, 0.72, "g/2 的理论与实验", fontsize=12, fontweight="bold", color=INK)
    ax2.text(0.02, 0.30, txt, fontsize=11, family="monospace", color=INK, linespacing=1.6)
    ax2.text(0.02, 0.10,
             "这是全部自然科学中理论与实验吻合得最好的一个数字——\n"
             "它的第一位来自狄拉克 1928 年那个「多余的」矩阵方程。",
             fontsize=9.5, color=INK_2, linespacing=1.5)
    return finish(fig, "05_g_minus_2.png",
                  "a_e = (g−2)/2 = α/2π − 0.328…(α/π)² + … 每一阶都是费曼图的贡献")


def main():
    section("1928 · 狄拉克 —— 相对论量子力学、反物质与自旋")
    print(__doc__)
    fig_dispersion()
    fig_zitterbewegung()
    fig_g_minus_2()


if __name__ == "__main__":
    use_style()
    main()
