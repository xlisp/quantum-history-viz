"""1982 费曼 · 1985 多伊奇 · 1994 肖尔 · 1996 格罗弗 —— 量子计算

费曼 1982《用计算机模拟物理》：「自然不是经典的。见鬼，如果你想模拟自然，
最好把计算机做成量子的。」  N 个量子比特的态要 2^N 个复数才能写下来，
经典计算机模拟量子系统天生指数级吃力 —— 那就反过来用量子系统来算。

  · 1985 多伊奇：通用量子图灵机；1992 Deutsch-Jozsa：第一个指数加速的（人造）问题
  · 1994 肖尔：大数分解的多项式算法（核心是量子傅里叶变换找周期）→ RSA 告急
  · 1996 格罗弗：无结构搜索 O(√N)，最优性可证
  · 2019 谷歌宣称「量子优越性」；2023–2025 逻辑量子比特与纠错成为主战场

本模块用 PyTorch 写了一个态矢量模拟器（qhv/qlib.py），然后：
  1) 量子比特的布洛赫球表示，以及 2^N 维态空间的指数爆炸；
  2) Deutsch-Jozsa：一次查询判定常数/平衡，经典最坏要 2^{n-1}+1 次；
  3) 格罗弗搜索：振幅放大的几何图像，π/4·√N 次迭代；
  4) 量子傅里叶变换找周期（肖尔算法的心脏）；
  5) VQE：把量子电路当成 PyTorch 模型，用 autograd + Adam 训练出多体基态能量。
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


# ------------------------------------------------------------ 1. 比特与指数爆炸
def fig_qubit_and_scaling():
    fig = plt.figure(figsize=(13.4, 5.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                    color=C["blue"], alpha=0.06, linewidth=0)
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                      color=INK_MUTED, alpha=0.18, linewidth=0.5, rstride=6, cstride=6)
    for vecs, col, lab in [((0, 0, 1), C["blue"], "|0〉"), ((0, 0, -1), C["blue"], "|1〉"),
                           ((1, 0, 0), C["orange"], "|+〉"), ((0, 1, 0), C["aqua"], "|+i〉")]:
        ax.quiver(0, 0, 0, *vecs, color=col, lw=2, arrow_length_ratio=0.12)
        ax.text(vecs[0] * 1.22, vecs[1] * 1.22, vecs[2] * 1.15, lab, fontsize=10, color=col)
    th, ph = 0.9, 0.9
    psi = (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th))
    ax.quiver(0, 0, 0, *psi, color=C["red"], lw=3, arrow_length_ratio=0.14)
    ax.text(psi[0] * 1.15, psi[1] * 1.15, psi[2] * 1.2,
            "|ψ〉= cos(θ/2)|0〉\n   + e^{iφ}sin(θ/2)|1〉", fontsize=9, color=C["red"])
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.set_title("① 一个量子比特 = 布洛赫球面上的一点\n（经典比特只能待在南北两极）",
                 fontsize=11.5)

    ax = fig.add_subplot(1, 2, 2)
    n = np.arange(1, 81)
    bytes_needed = 2.0 ** n * 16          # complex128 = 16 字节
    ax.semilogy(n, bytes_needed, color=C["blue"], lw=2.4, label="经典模拟所需内存 = 2^N × 16 字节")
    for label, size, col in [("一台笔记本 16 GB", 16 * 2 ** 30, C["orange"]),
                             ("一个大型超算 1 EB", 2 ** 60, C["aqua"]),
                             ("地球上全部硬盘 ~10²³ 字节", 1e23, C["violet"])]:
        ax.axhline(size, color=col, lw=1.6, ls="--")
        nn = math.log2(size / 16)
        ax.text(2, size * 1.8, f"{label}  → 只够存 {nn:.0f} 个量子比特",
                fontsize=9, color=col)
    ax.set_xlabel("量子比特数 N"); ax.set_ylabel("需要的经典内存（字节，对数轴）")
    ax.set_ylim(1e1, 1e30); ax.set_xlim(0, 80)
    ax.set_title("② 为什么要造量子计算机：2^N 的墙")
    ax.legend(loc="lower right", fontsize=9)
    print("  50 个量子比特的态矢量需要 %.1f PB 内存" % (2 ** 50 * 16 / 2 ** 50))
    return finish(fig, "09_qubit_bloch_scaling.png",
                  "注意：2^N 大 ≠ 量子计算无所不能——测量只能拿到 N 个比特的结果，算法必须靠干涉把答案「挤」到少数几个基态上")


# ------------------------------------------------------------ 2. Deutsch-Jozsa
def deutsch_jozsa(n: int, kind: str):
    """判定 f:{0,1}^n→{0,1} 是常数还是平衡的，只查询一次 oracle。"""
    psi = q.zero_state(n + 1)
    psi = q.apply1(psi, q.X, n)                      # 辅助比特置 |1〉
    for i in range(n + 1):
        psi = q.apply1(psi, q.H, i)
    # 相位 oracle：|x〉|y〉 → |x〉|y⊕f(x)〉，在 |−〉 上等价于打相位 (−1)^{f(x)}
    dim = 2 ** n
    xs = torch.arange(dim)
    if kind == "constant":
        fx = torch.zeros(dim, dtype=torch.long)
    else:
        # 随机平衡函数：一半输入映到 0，一半映到 1
        g = torch.Generator().manual_seed(5)
        perm = torch.randperm(dim, generator=g)
        fx = torch.zeros(dim, dtype=torch.long)
        fx[perm[: dim // 2]] = 1
    phase = torch.where(fx == 1, -1.0, 1.0).to(CT).reshape((2,) * n)
    psi = psi * phase.unsqueeze(-1)
    for i in range(n):
        psi = q.apply1(psi, q.H, i)
    p = q.probs(psi).reshape(dim, 2).sum(dim=1)
    return p


def qft_matrix(n: int) -> torch.Tensor:
    N = 2 ** n
    j = torch.arange(N)
    w = torch.exp(2j * math.pi * torch.outer(j, j).to(CT) / N)
    return w / math.sqrt(N)


def fig_dj_and_qft():
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8))

    # (a) Deutsch-Jozsa
    n = 4
    ax = axes[0]
    labels = q.bitstrings(n)
    pc = deutsch_jozsa(n, "constant").numpy()
    pb = deutsch_jozsa(n, "balanced").numpy()
    xpos = np.arange(2 ** n)
    ax.bar(xpos - 0.2, pc, width=0.4, color=C["blue"], label="常数函数 f ≡ 0")
    ax.bar(xpos + 0.2, pb, width=0.4, color=C["orange"], label="平衡函数（随机一半 0 一半 1）")
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("测量概率")
    ax.set_xlabel("第一寄存器的测量结果")
    ax.set_title("③ Deutsch-Jozsa：一次查询就分辨出来")
    ax.annotate("常数 → 100% 落在 |0000〉", (0, pc[0]), xytext=(2.6, 0.72),
                fontsize=9, color=C["blue"],
                arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1))
    ax.annotate(f"平衡 → |0000〉的概率严格为 0\n（这里是 {pb[0]:.0e}，纯属浮点误差）",
                (0.2, 0.0), xytext=(2.6, 0.40),
                fontsize=9, color=C["orange"],
                arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1))
    ax.text(0.5, -0.42, "经典确定性算法最坏需要 2^{n-1}+1 = 9 次查询；量子只要 1 次",
            transform=ax.transAxes, ha="center", fontsize=9, color=INK_2)
    ax.legend(loc="upper right", fontsize=8.5)
    print(f"  Deutsch-Jozsa：常数函数 P(|0…0〉) = {pc[0]:.3f}，平衡函数 P(|0…0〉) = {pb[0]:.3e}")

    # (b) QFT 找周期（肖尔算法的核心一步）
    nq, a, M = 9, 2, 21                       # 2^9=512 个 x；2^x mod 21 的周期 r=6
    N = 2 ** nq
    xs = np.arange(N)
    fx = np.array([pow(a, int(x), M) for x in xs])
    r = next(rr for rr in range(1, 30) if pow(a, rr, M) == 1)
    # 测量第二寄存器得到某个值 y0 → 第一寄存器坍缩成 {x : f(x)=y0} 的均匀叠加
    y0 = fx[3]
    mask = (fx == y0).astype(float)
    reg = torch.tensor(mask / math.sqrt(mask.sum()), dtype=CT)
    out = qft_matrix(nq) @ reg
    prob = (out.abs() ** 2).numpy()

    ax = axes[1]
    ax.plot(xs, prob, color=C["blue"], lw=1.6)
    ax.fill_between(xs, 0, prob, color=C["blue"], alpha=0.15)
    for k in range(r):
        pk = k * N / r
        ax.axvline(pk, color=C["orange"], lw=1.2, ls="--")
        if k:
            ax.text(pk + 4, prob.max() * 0.9, f"k·2ⁿ/r = {pk:.0f}", fontsize=8,
                    color=C["orange"], rotation=90, va="top")
    ax.set_ylim(0, prob.max() * 1.55)
    ax.set_xlabel("量子傅里叶变换输出（第一寄存器测量值）")
    ax.set_ylabel("概率")
    ax.set_title(f"④ QFT 找周期：峰值间距 = 2ⁿ/r（r = {r}）")
    ax.text(0.5, 0.97,
            f"a^x mod M：a={a}, M={M}\n"
            f"周期 r = {r}（量子部分只负责找出它）\n"
            f"再用连分数 + gcd 得到 M 的因子：\n"
            f"gcd(a^(r/2)±1, M) = {math.gcd(pow(a, r // 2, M) - 1, M)}, "
            f"{math.gcd(pow(a, r // 2, M) + 1, M)}",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.5", fc="#eef4fd", ec=C["blue"], lw=1),
            linespacing=1.5)
    print(f"  QFT 找周期：a={a}, M={M} → r={r}，因子 = "
          f"{math.gcd(pow(a, r//2, M)-1, M)} × {math.gcd(pow(a, r//2, M)+1, M)}")
    return finish(fig, "09_dj_qft.png",
                  "肖尔算法把「分解大数」化归为「找周期」，而找周期正是傅里叶变换最擅长的事——量子并行 + 干涉相消，答案自己浮出来")


# ------------------------------------------------------------ 3. Grover
def grover(n: int, marked: int, iters: int):
    """返回每一步迭代后的振幅分布。"""
    N = 2 ** n
    psi = q.zero_state(n)
    for i in range(n):
        psi = q.apply1(psi, q.H, i)
    amps = [q.vec(psi).real.clone()]
    for _ in range(iters):
        v = q.vec(psi).clone()
        v[marked] *= -1                                   # oracle：给目标打负号
        psi = v.reshape((2,) * n)
        for i in range(n):                                # 关于平均值反射（扩散算子）
            psi = q.apply1(psi, q.H, i)
        v = q.vec(psi).clone()
        v = -v
        v[0] *= -1
        psi = v.reshape((2,) * n)
        for i in range(n):
            psi = q.apply1(psi, q.H, i)
        amps.append(q.vec(psi).real.clone())
    return torch.stack(amps)


def fig_grover():
    n, marked = 10, 731
    N = 2 ** n
    opt = int(round(math.pi / 4 * math.sqrt(N)))
    amps = grover(n, marked, 2 * opt)
    probs = amps ** 2

    fig = plt.figure(figsize=(13.4, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.42, wspace=0.28)

    for j, it in enumerate([0, 4, opt]):
        ax = fig.add_subplot(gs[0, j])
        ax.bar(range(N), amps[it], width=1.0, color=C["blue"])
        ax.bar([marked], [amps[it][marked]], width=6, color=C["orange"])
        ax.set_ylim(-0.15, 1.05)
        ax.set_xlabel("基态编号 x")
        ax.set_ylabel("振幅 〈x|ψ〉" if j == 0 else "")
        ax.set_title(f"迭代 {it} 次   P(命中) = {probs[it][marked]:.3f}", fontsize=10.5)
        if j == 0:
            ax.annotate("均匀叠加\n每个振幅 1/√N", (N * 0.55, 0.10), xytext=(N * 0.3, 0.55),
                        fontsize=8.5, color=INK_2,
                        arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.9))
        if j == 2:
            ax.annotate("目标态振幅被放大到 ≈1", (marked, amps[it][marked]),
                        xytext=(N * 0.05, 0.72), fontsize=8.5, color=C["orange"],
                        arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1))

    ax = fig.add_subplot(gs[1, :2])
    it_axis = np.arange(len(probs))
    ax.plot(it_axis, probs[:, marked], color=C["blue"], lw=2.2, label="格罗弗：命中概率")
    theory = np.sin((2 * it_axis + 1) * math.asin(1 / math.sqrt(N))) ** 2
    ax.plot(it_axis, theory, ls="--", lw=1.8, color=INK_MUTED,
            label="解析 sin²((2k+1)·arcsin(1/√N))")
    ax.plot(it_axis, 1 - (1 - 1 / N) ** it_axis, color=C["orange"], lw=2,
            label="经典随机试探（同样的查询次数）")
    ax.axvline(opt, color=C["red"], ls=":", lw=1.6)
    ax.annotate(f"最优迭代次数 ≈ (π/4)√N = {opt}\n此时 P = {probs[opt][marked]:.4f}",
                xy=(opt, probs[opt][marked]), xytext=(opt + 4, 0.46),
                fontsize=9.5, color=C["red"],
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=1))
    ax.text(2 * opt - 2, 0.10, "继续转就转过头了\n（振幅是旋转，不是单调增长）",
            fontsize=8.8, color=INK_2, ha="right")
    ax.set_xlabel("格罗弗迭代次数 k")
    ax.set_ylabel("找到目标的概率")
    ax.set_title(f"⑤ 格罗弗搜索：N = {N} 个候选，只需 ~{opt} 次查询")
    ax.legend(loc="upper right", fontsize=8.5)

    ax = fig.add_subplot(gs[1, 2])
    ns = np.arange(4, 41)
    ax.semilogy(ns, 2.0 ** ns / 2, color=C["orange"], lw=2.2, label="经典 O(N) = 2^n/2")
    ax.semilogy(ns, math.pi / 4 * 2.0 ** (ns / 2), color=C["blue"], lw=2.2,
                label="格罗弗 O(√N)")
    ax.set_xlabel("数据库规模 n（比特）"); ax.set_ylabel("查询次数（对数轴）")
    ax.set_title("⑥ 平方加速")
    ax.legend(fontsize=8.5)
    ax.text(0.04, 0.72, "n=40：经典 5×10¹¹ 次\n格罗弗 8×10⁵ 次",
            transform=ax.transAxes, fontsize=9, color=INK_2)

    print(f"  格罗弗：N={N}，最优迭代 {opt} 次后命中概率 = {probs[opt][marked]:.4f}")
    return finish(fig, "09_grover.png",
                  "格罗弗迭代 = 二维平面内的旋转，每次转 2·arcsin(1/√N)；转到 90° 就该停手——这也是它的最优性证明")


# ------------------------------------------------------------ 4. VQE（PyTorch 训练量子电路）
def vqe_circuit(params: torch.Tensor, n: int, layers: int) -> torch.Tensor:
    """硬件高效 ansatz：每层 = 每个比特一个 Ry(θ) + 一圈 CNOT。全程可微。"""
    psi = q.zero_state(n)
    for i in range(n):
        psi = q.apply1(psi, q.H, i)
    for l in range(layers):
        for i in range(n):
            psi = q.apply1(psi, q.ry(params[l, i]), i)
        for i in range(n - 1):
            psi = q.cnot(psi, i, i + 1)
    for i in range(n):
        psi = q.apply1(psi, q.ry(params[layers, i]), i)
    return psi


def train_vqe(n=6, layers=3, steps=400, lr=0.08, seed=0):
    Hm = q.tfim_hamiltonian(n, J=1.0, g=1.0)
    e_exact = float(torch.linalg.eigvalsh(Hm)[0].real)
    g = torch.Generator().manual_seed(seed)
    params = (torch.rand((layers + 1, n), generator=g, dtype=DT) * 0.2).requires_grad_(True)
    opt = torch.optim.Adam([params], lr=lr)
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        psi = vqe_circuit(params, n, layers)
        v = q.vec(psi)
        energy = torch.vdot(v, Hm @ v).real
        energy.backward()
        opt.step()
        hist.append(float(energy))
    return np.array(hist), e_exact


def fig_vqe():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8))
    n = 6
    ramp = [SEQ_BLUE[i] for i in (3, 7, 10, 12)]
    finals = []
    for k, layers in enumerate([1, 2, 3, 5]):
        hist, e_exact = train_vqe(n=n, layers=layers)
        ax.plot(hist, color=ramp[k], lw=2, label=f"{layers} 层 ansatz（{(layers+1)*n} 个参数）")
        finals.append((layers, hist[-1], e_exact))
        print(f"  VQE  {layers} 层：E = {hist[-1]:.6f}，精确值 {e_exact:.6f}，"
              f"相对误差 {abs(hist[-1]/e_exact-1)*100:.3f}%")
    ax.axhline(e_exact, color=C["red"], lw=2, ls="--", label=f"精确基态能量 {e_exact:.4f}")
    ax.set_xlabel("Adam 迭代步"); ax.set_ylabel("能量期望值 〈ψ(θ)|Ĥ|ψ(θ)〉")
    ax.set_title(f"⑦ VQE：把量子电路当 PyTorch 模型来训练（TFIM, N={n}, g=J）")
    ax.legend(loc="upper right", fontsize=8.5)

    ax2.axis("off")
    ax2.text(0.0, 0.93, "变分量子本征求解器（VQE）在做什么", fontsize=12,
             fontweight="bold", color=INK)
    ax2.text(0.0, 0.10,
             "1.  参数化电路 U(θ)|0…0〉 生成试探态 |ψ(θ)〉\n"
             "2.  量子硬件负责测出能量 E(θ) = 〈ψ(θ)|Ĥ|ψ(θ)〉\n"
             "3.  经典优化器（这里是 Adam）沿 ∂E/∂θ 更新参数\n"
             "4.  变分原理保证 E(θ) ≥ E₀，永远不会「优化过头」\n\n"
             "本例的梯度直接由 PyTorch 的自动微分给出；真机上则用\n"
             "参数移位规则 ∂E/∂θ = [E(θ+π/2) − E(θ−π/2)] / 2 —— \n"
             "两次测量就能得到一个精确梯度，这是量子机器学习的基础。\n\n"
             "结果（N=6 横场 Ising，g=J，正处在量子临界点）：\n"
             + "\n".join(f"    {l} 层 → E = {e:.6f}   (精确 {ex:.6f}，误差 {abs(e/ex-1)*100:.3f}%)"
                         for l, e, ex in finals),
             fontsize=9.6, color=INK_2, linespacing=1.65, family="monospace")
    return finish(fig, "09_vqe.png",
                  "VQE / QAOA 属于「含噪中等规模量子（NISQ）」路线：浅电路 + 经典优化，是当下真机上最现实的算法家族")


def main():
    section("1982–今 · 量子计算 —— 费曼、多伊奇、肖尔、格罗弗")
    print(__doc__)
    fig_qubit_and_scaling()
    fig_dj_and_qft()
    fig_grover()
    fig_vqe()


if __name__ == "__main__":
    use_style()
    main()
