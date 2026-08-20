"""1984 BB84 · 1993 隐形传态 · 2016 墨子号 —— 量子通信

无克隆定理（Wootters-Zurek 1982）：不存在把任意未知量子态复制一份的幺正操作。
这是量子通信的两块基石之一：坏消息是不能中继放大，好消息是窃听必然留下痕迹。

  · 1984 Bennett-Brassard：BB84 量子密钥分发。安全性不靠计算复杂度，
    而靠物理定律 —— 窃听者测量就会扰动，扰动就会在误码率里暴露。
    截取-重发全拦截时误码率必然升到 25%。
  · 1993 Bennett 等：量子隐形传态。用 1 份纠缠 + 2 比特经典信息，
    把一个未知量子态搬到远处。注意「传的是态、不是物质」，且必须走经典信道，
    所以不违反相对论。
  · 1992 超密编码：反过来，1 个量子比特 + 1 份纠缠可以送 2 比特经典信息。
  · 2016 中国「墨子号」卫星：1200 km 纠缠分发、星地量子密钥分发；2017 京沪干线。

本模块用 qlib 的态矢量模拟器把这三件事都跑一遍，并给出可验证的数字。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import matplotlib.pyplot as plt

from . import qlib as q
from .style import use_style, finish, section, C, SERIES, SEQ_BLUE, INK, INK_2, INK_MUTED

CT = torch.complex128
DT = torch.float64


def random_qubit(gen) -> torch.Tensor:
    th = float(torch.rand(1, generator=gen, dtype=DT)) * math.pi
    ph = float(torch.rand(1, generator=gen, dtype=DT)) * 2 * math.pi
    return torch.tensor([math.cos(th / 2),
                         math.sin(th / 2) * (math.cos(ph) + 1j * math.sin(ph))], dtype=CT)


# ------------------------------------------------------------------ 隐形传态
def teleport(alpha: torch.Tensor):
    """返回 {(m0,m1): (概率, Bob 修正后的态)}。"""
    psi = torch.zeros((2, 2, 2), dtype=CT)
    psi[:, 0, 0] = alpha                       # q0 = 待传送的未知态
    psi = q.apply1(psi, q.H, 1)                # q1,q2 制备成贝尔对
    psi = q.cnot(psi, 1, 2)
    psi = q.cnot(psi, 0, 1)                    # Alice 的贝尔基测量
    psi = q.apply1(psi, q.H, 0)

    out = {}
    for m0 in (0, 1):
        for m1 in (0, 1):
            branch = psi[m0, m1, :]            # 投影到测量结果
            p = float((branch.abs() ** 2).sum())
            if p < 1e-12:
                out[(m0, m1)] = (0.0, None); continue
            bob = branch / math.sqrt(p)
            if m1:                             # Bob 按 2 比特经典信息做修正
                bob = q.X @ bob
            if m0:
                bob = q.Z @ bob
            out[(m0, m1)] = (p, bob)
    return out


def fig_teleportation(seed: int = 2):
    gen = torch.Generator().manual_seed(seed)
    fig = plt.figure(figsize=(13.4, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.95, 1.05], hspace=0.42, wspace=0.30)

    # 电路示意
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 10); ax.set_ylim(-0.6, 2.8); ax.axis("off")
    rows = {0: 2.4, 1: 1.4, 2: 0.3}
    names = ["Alice: |ψ〉未知态", "Alice: 纠缠对 A", "Bob:   纠缠对 B"]
    for r, y in rows.items():
        ax.plot([0.9, 9.4], [y, y], color=INK_MUTED, lw=1.2)
        ax.text(0.85, y, names[r], ha="right", va="center", fontsize=9.5, color=INK_2)

    def gate(x, y, label, col=C["blue"], w=0.42, h=0.30):
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, fc="white", ec=col, lw=1.6))
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5, color=col)

    def ctrl(x, y0, y1, col=C["aqua"]):
        ax.plot([x, x], [y0, y1], color=col, lw=1.6)
        ax.plot([x], [y0], "o", ms=7, color=col)
        ax.add_patch(plt.Circle((x, y1), 0.13, fc="white", ec=col, lw=1.6))
        ax.plot([x - 0.13, x + 0.13], [y1, y1], color=col, lw=1.4)
        ax.plot([x, x], [y1 - 0.13, y1 + 0.13], color=col, lw=1.4)

    gate(1.9, rows[1], "H")
    ctrl(2.6, rows[1], rows[2])
    ax.text(2.25, 2.75, "① 预先分发一对纠缠 |Φ⁺〉", fontsize=9.5, color=C["aqua"], ha="center")
    ctrl(4.0, rows[0], rows[1], col=C["blue"])
    gate(4.7, rows[0], "H")
    ax.text(4.35, 2.75, "② Alice 做贝尔基测量", fontsize=9.5, color=C["blue"], ha="center")
    for y in (rows[0], rows[1]):
        ax.add_patch(plt.Rectangle((5.3, y - 0.16), 0.42, 0.32, fc="#f4f4f2",
                                   ec=INK_2, lw=1.4))
        ax.text(5.51, y, "M", ha="center", va="center", fontsize=9.5, color=INK)
    ax.annotate("", xy=(8.0, rows[2] + 0.22), xytext=(5.8, rows[0] - 0.1),
                arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.6,
                                linestyle="--", connectionstyle="arc3,rad=-0.15"))
    ax.text(6.9, 1.75, "③ 2 比特经典信息\n（必须走经典信道，≤ 光速）",
            fontsize=9.5, color=C["orange"], ha="center")
    gate(8.4, rows[2], "X^m₁", C["red"], w=0.62)
    gate(9.15, rows[2], "Z^m₀", C["red"], w=0.62)
    ax.text(8.8, -0.35, "④ Bob 按经典比特做修正 → 得到 |ψ〉", fontsize=9.5,
            color=C["red"], ha="center")
    ax.set_title("量子隐形传态电路：1 份纠缠 + 2 比特经典信息 = 搬运 1 个未知量子态",
                 fontsize=12.5)

    # 保真度检验
    ax = fig.add_subplot(gs[1, 0])
    fids, probs_all = [], []
    for _ in range(400):
        a = random_qubit(gen)
        res = teleport(a)
        for (m0, m1), (p, bob) in res.items():
            probs_all.append(p)
            fids.append(float((torch.vdot(a, bob).abs() ** 2).real))
    ax.hist(fids, bins=40, range=(0.999, 1.001), color=C["blue"])
    ax.set_xlabel("保真度 |〈ψ|ψ_Bob〉|²")
    ax.set_ylabel("次数")
    ax.set_title("① 1600 次传送的保真度")
    ax.text(0.5, 0.6, f"最小保真度 = {min(fids):.12f}\n（误差全部来自浮点）",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc="#eef4fd", ec=C["blue"], lw=1))
    print(f"  隐形传态：400 个随机态 × 4 种测量结果，最小保真度 = {min(fids):.12f}")

    ax = fig.add_subplot(gs[1, 1])
    a = random_qubit(gen)
    res = teleport(a)
    keys = [f"m₀m₁ = {m0}{m1}" for (m0, m1) in res]
    ps = [res[k][0] for k in res]
    corr = ["I（什么都不做）", "X", "Z", "ZX"]
    ax.bar(range(4), ps, color=C["blue"], width=0.6)
    for i, (p, cstr) in enumerate(zip(ps, corr)):
        ax.text(i, p + 0.012, f"{p:.3f}\n{cstr}", ha="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(range(4)); ax.set_xticklabels(keys, fontsize=8.5)
    ax.set_ylim(0, 0.36)
    ax.set_title("② 四种测量结果等概率出现")
    ax.text(0.5, 0.97, "Alice 的结果本身完全随机 →\n单看 Bob 那边什么信息也没有；\n"
                       "没有那 2 比特经典信息，\nBob 手里就是最大混合态",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.6,
            color=INK_2, linespacing=1.5)

    # 超密编码
    ax = fig.add_subplot(gs[1, 2])
    dec = np.zeros((4, 4))
    for idx, (msg, gate_op) in enumerate([("00", q.I2), ("01", q.X), ("10", q.Z),
                                          ("11", q.Z @ q.X)]):
        psi = q.zero_state(2)
        psi = q.apply1(psi, q.H, 0)
        psi = q.cnot(psi, 0, 1)               # 共享纠缠
        psi = q.apply1(psi, gate_op, 0)       # Alice 只动自己那个比特
        psi = q.cnot(psi, 0, 1)               # Bob 解码
        psi = q.apply1(psi, q.H, 0)
        dec[idx] = q.probs(psi).numpy()
    im = ax.imshow(dec, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(q.bitstrings(2))
    ax.set_yticks(range(4)); ax.set_yticklabels(["00", "01", "10", "11"])
    ax.set_xlabel("Bob 的测量结果"); ax.set_ylabel("Alice 想发的 2 比特")
    ax.set_title("③ 超密编码：1 个量子比特送 2 比特")
    for i in range(4):
        for j in range(4):
            if dec[i, j] > 0.01:
                ax.text(j, i, f"{dec[i,j]:.0f}", ha="center", va="center",
                        color="white", fontsize=10)
    ax.grid(False)
    print(f"  超密编码：解码矩阵 = 单位阵，最小对角元 = {dec.diagonal().min():.6f}")
    return finish(fig, "09_teleportation.png",
                  "隐形传态传的是「态」不是物质，原件在测量中被销毁（与无克隆定理自洽）；必须等经典信息到达才能重建，所以不能超光速")


# ------------------------------------------------------------------ BB84
def bb84(n_bits: int, eve_frac: float, gen) -> tuple[float, int]:
    """模拟 BB84：返回 (筛选后误码率 QBER, 筛选后密钥长度)。"""
    a_bits = torch.randint(0, 2, (n_bits,), generator=gen)
    a_base = torch.randint(0, 2, (n_bits,), generator=gen)     # 0=直角基 1=对角基
    b_base = torch.randint(0, 2, (n_bits,), generator=gen)
    intercepted = torch.rand(n_bits, generator=gen, dtype=DT) < eve_frac
    e_base = torch.randint(0, 2, (n_bits,), generator=gen)

    bits = a_bits.clone()
    base_now = a_base.clone()
    # Eve 截取-重发：基选对了不留痕迹，选错了她的结果是随机的，并把光子重新制备到自己的基上
    wrong_e = intercepted & (e_base != a_base)
    rnd = torch.randint(0, 2, (n_bits,), generator=gen)
    bits = torch.where(wrong_e, rnd, bits)
    base_now = torch.where(intercepted, e_base, base_now)

    # Bob 测量：基一致则忠实读出，否则 50/50
    wrong_b = b_base != base_now
    rnd2 = torch.randint(0, 2, (n_bits,), generator=gen)
    b_bits = torch.where(wrong_b, rnd2, bits)

    keep = a_base == b_base                                     # 公开比对基，筛选
    sifted_a, sifted_b = a_bits[keep], b_bits[keep]
    if sifted_a.numel() == 0:
        return 0.0, 0
    qber = float((sifted_a != sifted_b).double().mean())
    return qber, int(sifted_a.numel())


def h2(x):
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def fig_bb84(seed: int = 4):
    gen = torch.Generator().manual_seed(seed)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))

    # ① 协议本身：无窃听 vs 全窃听
    ax = axes[0]
    for eve, col, lab in [(0.0, C["blue"], "无窃听"), (1.0, C["red"], "Eve 全程截取-重发")]:
        qbers = [bb84(4000, eve, gen)[0] for _ in range(60)]
        ax.hist(qbers, bins=np.linspace(-0.005, 0.32, 80), color=col, alpha=0.85,
                label=f"{lab}：QBER ≈ {np.mean(qbers):.3f}")
        print(f"  BB84 {lab}：QBER = {np.mean(qbers):.4f} ± {np.std(qbers):.4f}")
    ax.axvline(0.25, color=INK_MUTED, ls="--", lw=1.4)
    ax.set_xlim(-0.01, 0.32)
    ax.text(0.25, ax.get_ylim()[1] * 0.55, "  理论值 25%", fontsize=9, color=INK_2)
    ax.set_xlabel("筛选后误码率 QBER"); ax.set_ylabel("模拟轮次")
    ax.set_title("① 窃听一定会留下痕迹")
    ax.legend(fontsize=8.5)

    # ② QBER 随窃听比例线性上升
    ax = axes[1]
    fr = np.linspace(0, 1, 21)
    meas = [np.mean([bb84(6000, float(f), gen)[0] for _ in range(8)]) for f in fr]
    ax.plot(fr, meas, "o", ms=6, color=C["orange"], label="蒙特卡洛模拟")
    ax.plot(fr, fr / 4, color=C["blue"], lw=2, label="理论 QBER = η/4")
    ax.axhline(0.11, color=C["red"], ls="--", lw=1.8)
    ax.text(0.02, 0.118, "BB84 安全阈值 11%（Shor-Preskill）", fontsize=8.8, color=C["red"])
    ax.set_xlabel("Eve 拦截的光子比例 η"); ax.set_ylabel("QBER")
    ax.set_title("② 窃听得越多，暴露得越彻底")
    ax.legend(loc="upper left", fontsize=8.5)

    # ③ 安全密钥率
    ax = axes[2]
    e = np.linspace(0, 0.16, 300)
    rate = np.clip(1 - 2 * h2(e), 0, None)
    ax.plot(e * 100, rate, color=C["blue"], lw=2.4, label="安全密钥率 r = 1 − 2h(e)")
    ax.fill_between(e * 100, 0, rate, color=C["blue"], alpha=0.10)
    ax.axvline(11.0, color=C["red"], ls="--", lw=1.8)
    ax.text(11.4, 0.88, "e > 11% → 密钥率归零\n协议中止，本轮全部丢弃",
            fontsize=9, color=C["red"], va="top")
    ax.set_xlabel("误码率 e (%)"); ax.set_ylabel("每个筛选比特能提取的安全密钥")
    ax.set_ylim(0, 1.05)
    ax.set_title("③ 安全性可以被定量证明")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.text(0.97, 0.42, "BB84 的安全性不依赖\n「敌人算力不够」这种假设，\n"
                        "而依赖无克隆定理与测量扰动 ——\n量子计算机再强也破不了。",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.8,
            color=INK_2, linespacing=1.5)
    return finish(fig, "09_bb84_qkd.png",
                  "截取-重发时 Eve 有 1/2 概率选错基，选错时 Bob 又有 1/2 概率读错 → 筛选后误码率 = η/4，全拦截即 25%")


def main():
    section("1984–今 · 量子通信 —— 密钥分发、隐形传态、超密编码")
    print(__doc__)
    fig_teleportation()
    fig_bb84()


if __name__ == "__main__":
    use_style()
    main()
