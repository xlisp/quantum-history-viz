"""1900 · 普朗克：一次「绝望的行为」，量子诞生

19 世纪末，经典物理只剩「两朵乌云」。其中一朵就是黑体辐射：
  · 瑞利-金斯定律（经典能均分定理）  B_λ = 2ckT / λ⁴     → λ→0 时发散：紫外灾变
  · 维恩位移律给出的近似只在短波成立

1900-12-14，普朗克在柏林报告：只要假设振子的能量只能取 E = nhν 这样一份一份的
「能量子」，就能得到与实验完全吻合的公式

  B_λ(λ,T) = (2hc²/λ⁵) · 1/(exp(hc/λk_BT) − 1)

本模块做三件事：
  1) 把普朗克公式、瑞利-金斯、维恩近似画在一起，直观看到「紫外灾变」；
  2) 用 PyTorch 的自动微分，从带噪声的"实验光谱"里反演出 h 和 k_B
     （这正是普朗克本人 1900 年做的事：他由此第一次给出了 h 和玻尔兹曼常数）；
  3) 画出单个振子的平均能量 <E> = hν/(exp(hν/kT)−1)，看高频模式如何被「冻结」。
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt

from .style import use_style, finish, section, C, SERIES, INK_2, INK_MUTED

# --- CODATA 2018 真值 ---
H_TRUE = 6.62607015e-34      # J·s
K_TRUE = 1.380649e-23        # J/K
C_LIGHT = 2.99792458e8       # m/s
WIEN_B = 2.897771955e-3      # m·K


def planck_lambda(lam, T, h=H_TRUE, k=K_TRUE):
    """谱辐射亮度 B_λ，单位 W·m⁻²·sr⁻¹·m⁻¹。lam/T 可为 torch 张量。"""
    x = h * C_LIGHT / (lam * k * T)
    return 2 * h * C_LIGHT ** 2 / lam ** 5 / torch.expm1(x)


def rayleigh_jeans(lam, T, k=K_TRUE):
    return 2 * C_LIGHT * k * T / lam ** 4


def wien_approx(lam, T, h=H_TRUE, k=K_TRUE):
    return 2 * h * C_LIGHT ** 2 / lam ** 5 * torch.exp(-h * C_LIGHT / (lam * k * T))


# ------------------------------------------------------------------ 图 1
def fig_uv_catastrophe():
    lam = torch.logspace(-7.3, -4.7, 800, dtype=torch.float64)   # 50nm ~ 20µm
    temps = [3000.0, 4500.0, 5800.0]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for i, T in enumerate(temps):
        B = planck_lambda(lam, T)
        ax.loglog(lam * 1e6, B, color=SERIES[i], lw=2,
                  label=f"普朗克定律 T={T:.0f} K")
        lmax = WIEN_B / T
        Bmax = planck_lambda(torch.tensor(lmax, dtype=torch.float64), T)
        ax.plot(lmax * 1e6, Bmax, "o", color=SERIES[i], ms=8,
                markeredgecolor="white", markeredgewidth=1.6, zorder=5)
        ax.annotate(f"λ_max={lmax*1e6:.2f} µm", (lmax * 1e6, Bmax),
                    textcoords="offset points", xytext=(8, 10),
                    color=SERIES[i], fontsize=9)
    RJ = rayleigh_jeans(lam, 5800.0)
    ax.loglog(lam * 1e6, RJ, color=C["red"], lw=2, ls="--",
              label="瑞利-金斯（经典）T=5800 K")
    W = wien_approx(lam, 5800.0)
    ax.loglog(lam * 1e6, W, color=C["violet"], lw=1.6, ls=":",
              label="维恩近似 T=5800 K")

    ax.fill_betweenx([1e2, 1e16], 0.05, 0.4, color=C["red"], alpha=0.06)
    ax.text(0.09, 3e14, "紫外灾变区\n经典理论 →∞", color=C["red"], fontsize=10,
            ha="left", va="top", fontweight="bold")
    ax.set_xlim(0.05, 20)
    ax.set_ylim(1e2, 1e15)
    ax.set_xlabel("波长 λ (µm)")
    ax.set_ylabel("谱辐射亮度 B_λ (W·m⁻²·sr⁻¹·m⁻¹)")
    ax.set_title("黑体辐射：经典物理的崩塌与量子假设的胜利")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.35)
    return finish(fig, "01_planck_uv_catastrophe.png",
                  "维恩位移律 λ_max·T = 2.898×10⁻³ m·K；瑞利-金斯在 λ→0 发散，普朗克公式则回落到零")


# ------------------------------------------------------------------ 图 2
def fit_constants(seed: int = 0, steps: int = 4000):
    """用 PyTorch autograd 从"实验光谱"里把 h 和 k_B 拟合出来。"""
    g = torch.Generator().manual_seed(seed)
    temps = torch.tensor([1000.0, 1200.0, 1400.0, 1600.0], dtype=torch.float64)
    lam = torch.logspace(np.log10(8e-7), np.log10(2e-5), 60, dtype=torch.float64)
    L, T = torch.meshgrid(lam, temps, indexing="ij")
    clean = planck_lambda(L, T)
    noise = 1 + 0.03 * torch.randn(clean.shape, generator=g, dtype=torch.float64)
    data = clean * noise                              # 3% 乘性噪声 = "实验数据"

    # 以真值的 40% / 250% 作为很糟糕的初始猜测，在对数空间里优化保证正定
    log_h = torch.tensor(np.log(H_TRUE * 0.4), dtype=torch.float64, requires_grad=True)
    log_k = torch.tensor(np.log(K_TRUE * 2.5), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_h, log_k], lr=0.02)

    hist = []
    for step in range(steps):
        opt.zero_grad()
        pred = planck_lambda(L, T, h=torch.exp(log_h), k=torch.exp(log_k))
        loss = ((torch.log(pred) - torch.log(data)) ** 2).mean()
        loss.backward()
        opt.step()
        hist.append((float(loss), float(torch.exp(log_h)), float(torch.exp(log_k))))
    return np.array(hist), lam, temps, data, clean


def fig_fit():
    hist, lam, temps, data, clean = fit_constants()
    h_fit, k_fit = hist[-1, 1], hist[-1, 2]
    print(f"  拟合得到 h = {h_fit:.5e} J·s   (真值 {H_TRUE:.5e}，误差 {abs(h_fit/H_TRUE-1)*100:.2f}%)")
    print(f"  拟合得到 k = {k_fit:.5e} J/K   (真值 {K_TRUE:.5e}，误差 {abs(k_fit/K_TRUE-1)*100:.2f}%)")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    ax = axes[0]
    for i, T in enumerate(temps):
        ax.plot(lam * 1e6, data[:, i], "o", ms=3.5, color=SERIES[i], alpha=0.55)
        ax.plot(lam * 1e6, planck_lambda(lam, float(T), h_fit, k_fit),
                color=SERIES[i], lw=2, label=f"T={float(T):.0f} K")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("λ (µm)")
    ax.set_ylabel("B_λ")
    ax.set_title("① 合成「实验」光谱与拟合曲线")
    ax.legend()

    ax = axes[1]
    ax.semilogy(hist[:, 0], color=C["blue"], lw=2)
    ax.set_xlabel("Adam 迭代步")
    ax.set_ylabel("对数残差均方 loss")
    ax.set_title("② 梯度下降收敛")

    ax = axes[2]
    ax.plot(hist[:, 1] / H_TRUE, color=C["blue"], lw=2, label="h / h_真值")
    ax.plot(hist[:, 2] / K_TRUE, color=C["orange"], lw=2, label="k_B / k_真值")
    ax.axhline(1.0, color=INK_MUTED, lw=1, ls="--")
    ax.set_ylim(0, 3)
    ax.set_xlabel("Adam 迭代步")
    ax.set_ylabel("与真值之比")
    ax.set_title("③ 两个自然常数被「反演」出来")
    ax.annotate(f"h={h_fit:.4e}", (len(hist) * 0.55, 1.12), color=C["blue"], fontsize=9)
    ax.annotate(f"k={k_fit:.4e}", (len(hist) * 0.55, 0.72), color=C["orange"], fontsize=9)
    ax.legend(loc="upper right")

    return finish(fig, "01_planck_fit_constants.png",
                  "把普朗克公式当成 PyTorch 里的可微模型，对 log h、log k 做 Adam 优化——1900 年普朗克正是靠拟合光谱首次定出了 h 与 k_B")


# ------------------------------------------------------------------ 图 3
def fig_mean_energy():
    T = 300.0
    x = torch.logspace(-2, 1.4, 500, dtype=torch.float64)   # x = hν/kT
    quantum = x / torch.expm1(x)          # <E>/kT ，普朗克
    classical = torch.ones_like(x)        # 能均分定理：每个模式 kT

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.semilogx(x, classical, color=C["red"], ls="--", lw=2,
                label="经典能均分：⟨E⟩ = k_BT")
    ax.semilogx(x, quantum, color=C["blue"], lw=2.4,
                label="普朗克振子：⟨E⟩ = hν/(e^{hν/k_BT} − 1)")
    ax.fill_between(x.numpy(), quantum.numpy(), classical.numpy(),
                    color=C["blue"], alpha=0.08)
    ax.axvline(1.0, color=INK_MUTED, lw=1, ls=":")
    ax.text(1.08, 0.55, "hν ≈ k_BT\n高频模式开始被「冻结」", fontsize=9, color=INK_2)
    ax.set_xlabel("x = hν / k_BT  （频率↑ 或 温度↓）")
    ax.set_ylabel("⟨E⟩ / k_BT")
    ax.set_ylim(0, 1.15)
    ax.set_title("能量为什么必须「一份一份」：高频模式被冻结")
    ax.legend(loc="lower left")
    return finish(fig, "01_planck_mean_energy.png",
                  "经典下每个自由度都分到 k_BT（于是模式无穷多 → 能量无穷）；能量子假设让 hν ≫ k_BT 的模式激发不起来")


def main():
    section("1900 · 普朗克 —— 能量子假设  E = hν")
    print(__doc__)
    fig_uv_catastrophe()
    fig_fit()
    fig_mean_energy()


if __name__ == "__main__":
    use_style()
    main()
