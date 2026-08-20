"""1925 矩阵力学 · 1927 不确定性原理 —— 海森堡

海森堡在赫尔戈兰岛（1925 夏）想通了一件事：不要去问电子的「轨道」，
只谈可观测量（谱线的频率与强度）。于是物理量变成了矩阵，而矩阵不对易：

    [x̂, p̂] = x̂p̂ − p̂x̂ = iħ          （玻恩-约当把它写成了正则对易关系）

由此直接推出（1927）：
    σ_x · σ_p ≥ ħ/2                    海森堡不确定性原理
更一般地（罗伯逊-薛定谔不等式）：σ_A σ_B ≥ ½|〈[Â,B̂]〉|

注意：不确定性不是「测不准」（仪器不好），而是「本来就没有同时确定的值」——
x 和 p 的本征基彼此是傅里叶变换，一个窄，另一个必然宽。

本模块：
  1) 用产生/湮灭算符 a†、a 直接把 x̂、p̂、Ĥ 造成矩阵，验证 [x̂,p̂]=iħ、E_n=(n+½)ħω；
  2) 数值验证 σ_xσ_p：Fock 态给出 (n+½)ħ，相干态取到下限 ħ/2，压缩态则一头小一头大；
  3) 自由高斯波包随时间扩散：σ_x 变大而 σ_p 不变，乘积单调增长。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .style import use_style, finish, section, C, SERIES, INK, INK_2, INK_MUTED, SEQ_BLUE

DT, CT = torch.float64, torch.complex128
# 发散色标：蓝 ↔ 红，中点为中性灰（绝不用彩虹色）
DIVERGING = LinearSegmentedColormap.from_list(
    "bluered", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#8f1b1a"])


def ladder_ops(N: int = 24):
    """截断到 N 维的产生/湮灭算符（ħ = m = ω = 1）。"""
    n = torch.arange(1, N, dtype=DT)
    a = torch.diag(torch.sqrt(n), 1).to(CT)          # a|n> = √n |n-1>
    ad = a.conj().T
    x = (a + ad) / math.sqrt(2)
    p = 1j * (ad - a) / math.sqrt(2)
    Hm = ad @ a + 0.5 * torch.eye(N, dtype=CT)
    return a, ad, x, p, Hm


def fig_matrix_mechanics():
    N = 12
    a, ad, x, p, Hm = ladder_ops(N)
    comm = (x @ p - p @ x)                            # 应当 = i·I（除最后一行截断误差）
    print(f"  [x,p] 的对角元 = {complex(comm[0,0]):.3f}, {complex(comm[3,3]):.3f} …（理论值 1i）")
    print(f"  Ĥ 的本征值 = {[round(float(v), 3) for v in torch.linalg.eigvalsh(Hm)[:6]]}  (理论 n+1/2)")

    mats = [(x.real, "位置算符 x̂ = (a+a†)/√2", DIVERGING),
            (p.imag, "动量算符 p̂ 的虚部 Im(p̂)", DIVERGING),
            ((comm / 1j).real, "对易子 [x̂,p̂]/i = ħ·I", DIVERGING),
            (Hm.real, "哈密顿量 Ĥ = a†a + ½", DIVERGING)]
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.9))
    for ax, (M, title, cmap) in zip(axes, mats):
        M = M.numpy()
        v = max(abs(M).max(), 1e-9)
        im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -v, v))
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks(range(0, N, 3)); ax.set_yticks(range(0, N, 3))
        ax.set_xlabel("列 n'"); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("行 n")
    axes[2].text(0.5, -0.30, "只有对角元非零 → 处处 = iħ\n（右下角是截断造成的假象）",
                 transform=axes[2].transAxes, ha="center", fontsize=8.5, color=INK_MUTED)
    fig.suptitle("矩阵力学：可观测量就是矩阵，它们一般不对易", fontsize=13,
                 fontweight="bold", y=1.06)
    return finish(fig, "04_matrix_mechanics.png",
                  "海森堡 1925 年重建力学时并不知道自己写的是矩阵；玻恩认出来后与约当一起写下 [x,p]=iħ —— 量子力学最核心的一行公式")


def sigma_xp(psi_x: torch.Tensor, x: torch.Tensor):
    """给定实空间波函数，用 FFT 得到动量分布，返回 (σ_x, σ_p)。"""
    dx = float(x[1] - x[0])
    rho = psi_x.abs() ** 2
    rho = rho / (rho.sum() * dx)
    mx = (x * rho).sum() * dx
    sx = torch.sqrt(((x - mx) ** 2 * rho).sum() * dx)

    n = x.numel()
    k = 2 * math.pi * torch.fft.fftfreq(n, d=dx).to(DT)
    phi = torch.fft.fft(psi_x)
    rk = phi.abs() ** 2
    k, rk = torch.fft.fftshift(k), torch.fft.fftshift(rk)
    dk = float(k[1] - k[0])
    rk = rk / (rk.sum() * dk)
    mk = (k * rk).sum() * dk
    sp = torch.sqrt(((k - mk) ** 2 * rk).sum() * dk)          # ħ = 1 → p = k
    return float(sx), float(sp)


def fig_uncertainty():
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))

    # (a) Fock 态：σ_xσ_p = n + 1/2
    N = 40
    a, ad, x_op, p_op, Hm = ladder_ops(N)
    ns, prods = [], []
    for n in range(8):
        ket = torch.zeros(N, dtype=CT); ket[n] = 1
        x2 = float((torch.vdot(ket, (x_op @ x_op) @ ket)).real)
        p2 = float((torch.vdot(ket, (p_op @ p_op) @ ket)).real)
        ns.append(n); prods.append(math.sqrt(x2 * p2))
    ax = axes[0]
    ax.bar(ns, prods, color=C["blue"], width=0.62)
    ax.plot(ns, [n + 0.5 for n in ns], "o--", color=C["orange"], lw=1.6, ms=6,
            label="理论 (n + ½)ħ")
    ax.axhline(0.5, color=C["red"], lw=1.8, ls="--")
    ax.text(7.42, 0.62, "量子极限 ħ/2\n谁也下不去", color=C["red"], fontsize=9,
            ha="right", va="bottom")
    for n, v in zip(ns, prods):
        ax.text(n, v + 0.16, f"{v:.1f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Fock 态 |n〉"); ax.set_ylabel("σ_x · σ_p  (ħ = 1)")
    ax.set_title("① 能量越高，相空间「面积」越大")
    ax.legend(loc="upper left")

    # (b) 高斯波包：宽度扫描，乘积恒为 1/2（最小不确定态）
    xg = torch.linspace(-40, 40, 8192, dtype=DT)
    widths = np.linspace(0.35, 6.0, 24)
    sx_l, sp_l, pr_l = [], [], []
    for w in widths:
        psi = torch.exp(-xg ** 2 / (2 * w ** 2)).to(CT)
        sx, sp = sigma_xp(psi, xg)
        sx_l.append(sx); sp_l.append(sp); pr_l.append(sx * sp)
    ax = axes[1]
    ax.plot(sx_l, sp_l, "o-", color=C["blue"], ms=5, label="高斯波包（最小不确定态）")
    sxx = np.linspace(0.2, 4.5, 200)
    ax.plot(sxx, 0.5 / sxx, color=C["red"], lw=2, ls="--", label="σ_xσ_p = ħ/2 双曲线")
    ax.fill_between(sxx, 0, 0.5 / sxx, color=C["red"], alpha=0.07)
    ax.text(2.05, 0.66, "禁区：σ_xσ_p < ħ/2\n物理上不存在", color=C["red"], fontsize=9)
    ax.set_xlim(0, 4.5); ax.set_ylim(0, 1.6)
    ax.set_xlabel("σ_x"); ax.set_ylabel("σ_p")
    ax.set_title("② 位置越确定，动量越不确定")
    ax.legend(loc="upper right")
    print(f"  高斯波包 σ_xσ_p = {np.mean(pr_l):.4f} ± {np.std(pr_l):.1e}（理论 0.5）")

    # (c) 自由波包扩散：σ_x(t) 变大，σ_p 不变
    xg2 = torch.linspace(-120, 120, 8192, dtype=DT)
    dx = float(xg2[1] - xg2[0])
    k = 2 * math.pi * torch.fft.fftfreq(xg2.numel(), d=dx).to(DT)
    s0 = 2.0
    psi = torch.exp(-xg2 ** 2 / (2 * s0 ** 2)).to(CT)
    ts, sxs, sps = [], [], []
    dt = 0.25
    for step in range(121):
        if step % 4 == 0:
            sx, sp = sigma_xp(psi, xg2)
            ts.append(step * dt); sxs.append(sx); sps.append(sp)
        psi = torch.fft.ifft(torch.exp(-0.5j * (k ** 2).to(CT) * dt) * torch.fft.fft(psi))
    ax = axes[2]
    ax.plot(ts, sxs, color=C["blue"], lw=2, label="σ_x(t) 位置弥散")
    ax.plot(ts, sps, color=C["orange"], lw=2, label="σ_p(t) 动量弥散（守恒）")
    ax.plot(ts, np.array(sxs) * np.array(sps), color=C["violet"], lw=2, label="乘积 σ_xσ_p")
    theory = (s0 / math.sqrt(2)) * np.sqrt(1 + (np.array(ts) / s0 ** 2) ** 2)
    ax.plot(ts, theory, ls=":", lw=2, color=INK_MUTED, label="解析 σ_x(t)")
    ax.axhline(0.5, color=C["red"], lw=1.5, ls="--")
    ax.set_xlabel("时间 t"); ax.set_ylabel("弥散")
    ax.set_title("③ 自由粒子波包必然扩散")
    ax.legend(loc="upper left", fontsize=8.5)

    return finish(fig, "04_uncertainty.png",
                  "σ_x(t) = σ_0√(1+(ħt/2mσ_0²)²)：初始越「聚焦」，散得越快 —— 这正是显微镜分辨率与量子极限的来源")


def main():
    section("1925–1927 · 海森堡 —— 矩阵力学与不确定性原理")
    print(__doc__)
    fig_matrix_mechanics()
    fig_uncertainty()


if __name__ == "__main__":
    use_style()
    main()
