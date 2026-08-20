"""1935 EPR · 1964 贝尔 · 1982 阿斯佩 · 2022 诺贝尔 —— 量子纠缠

1935 年 EPR 论文《能认为量子力学对物理实在的描述是完备的吗？》：
若两个粒子处在纠缠态，测 A 立刻决定 B 的结果，爱因斯坦称之为
「幽灵般的超距作用」，并主张量子力学不完备，应该存在隐变量。

    |Φ⁺〉 = (|00〉+|11〉)/√2       —— 无法写成 |a〉⊗|b〉 的态就叫纠缠态

薛定谔同年造出 「Verschränkung / entanglement」 这个词，并说：
「纠缠不是量子力学的一个特征，而是它的那个特征。」

1964 年贝尔把哲学争论变成了可做的实验：任何「定域实在论」（局域隐变量）模型都必须满足

    |S| = |E(a,b) − E(a,b') + E(a',b) + E(a',b')| ≤ 2        （CHSH 不等式）

而量子力学允许 S = 2√2 ≈ 2.828（西雷尔森上界）。
1972 起 Freedman-Clauser、1982 阿斯佩、2015 三个无漏洞实验依次判定：
自然界站在量子力学一边。阿斯佩、克劳泽、蔡林格获 2022 年诺贝尔物理学奖。

本模块用 PyTorch 完成：
  1) 纠缠的度量：约化密度矩阵与冯·诺依曼熵 S = −Tr ρ_A log₂ ρ_A；
  2) 关联函数 E(a,b) = cos(a−b)：量子 vs 局域隐变量模型；
  3) CHSH 扫描 + 有限次数抽样实验，看到 S 超过 2 达数十个标准差；
  4) 验证「纠缠不能传信」：Alice 的边缘概率完全不依赖 Bob 的测量方向。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from . import qlib as q
from .style import use_style, finish, section, C, SERIES, INK, INK_2, INK_MUTED
from .s04_heisenberg import DIVERGING

CT = torch.complex128


def bell_states():
    """用 H + CNOT 电路生成四个贝尔态。"""
    out = {}
    for name, bits in [("Φ⁺", (0, 0)), ("Φ⁻", (1, 0)), ("Ψ⁺", (0, 1)), ("Ψ⁻", (1, 1))]:
        psi = q.basis_state(bits)
        psi = q.apply1(psi, q.H, 0)
        psi = q.cnot(psi, 0, 1)
        out[name] = psi
    return out


def fig_bell_and_entropy():
    bs = bell_states()
    fig = plt.figure(figsize=(13.6, 7.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1], hspace=0.45, wspace=0.35)

    labels = q.bitstrings(2)
    for i, (name, psi) in enumerate(bs.items()):
        rho = q.density(psi).real.numpy()
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(rho, cmap=DIVERGING, norm=TwoSlopeNorm(0, -0.5, 0.5))
        for r in range(4):
            for c in range(4):
                if abs(rho[r, c]) > 1e-9:
                    ax.text(c, r, f"{rho[r,c]:+.1f}", ha="center", va="center",
                            fontsize=8, color=INK)
        S = q.entropy_vn(q.partial_trace_pure(psi, [0]))
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(labels, fontsize=7.5); ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_title(f"|{name}〉   S = {S:.2f} ebit", fontsize=10.5)
        ax.grid(False)
    fig.text(0.5, 0.945, "四个贝尔态的密度矩阵 ρ = |ψ〉〈ψ|：对角外的元素就是相干性",
             ha="center", fontsize=12.5, fontweight="bold")

    # 纠缠度随参数变化： |ψ(θ)〉 = cosθ|00〉 + sinθ|11〉
    ax = fig.add_subplot(gs[1, :2])
    th = np.linspace(0, math.pi / 2, 200)
    S_list, conc = [], []
    for t in th:
        psi = torch.zeros((2, 2), dtype=CT)
        psi[0, 0] = math.cos(t); psi[1, 1] = math.sin(t)
        S_list.append(q.entropy_vn(q.partial_trace_pure(psi, [0])))
        conc.append(abs(math.sin(2 * t)))                 # concurrence
    ax.plot(np.degrees(th), S_list, color=C["blue"], lw=2.4, label="纠缠熵 S (ebit)")
    ax.plot(np.degrees(th), conc, color=C["orange"], lw=2, ls="--", label="并发度 C = |sin2θ|")
    ax.axvline(45, color=INK_MUTED, ls=":", lw=1)
    ax.plot([45], [1.0], "o", ms=9, color=C["blue"], markeredgecolor="white", mew=1.6)
    ax.annotate("θ=45°：最大纠缠\n|Φ⁺〉，S = 1 ebit", (45, 1.0),
                xytext=(50, 0.42), fontsize=9.5, color=C["blue"],
                arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1))
    ax.text(2, 0.60, "θ=0：|00〉可分离\nS = 0（无纠缠）", fontsize=9, color=INK_2)
    ax.set_xlabel("θ (度)   |ψ(θ)〉 = cosθ|00〉 + sinθ|11〉")
    ax.set_ylabel("纠缠度")
    ax.set_title("纠缠是可以定量的：从可分离到最大纠缠")
    ax.legend(loc="lower right")

    # 纯度：整体是纯态，局部却是最大混合态
    ax = fig.add_subplot(gs[1, 2:])
    names = ["整体 ρ_AB", "局部 ρ_A", "局部 ρ_B"]
    psi = bs["Φ⁺"]
    vals = [q.purity(q.density(psi)),
            q.purity(q.partial_trace_pure(psi, [0])),
            q.purity(q.partial_trace_pure(psi, [1]))]
    ent = [q.entropy_vn(q.density(psi)),
           q.entropy_vn(q.partial_trace_pure(psi, [0])),
           q.entropy_vn(q.partial_trace_pure(psi, [1]))]
    xpos = np.arange(3)
    ax.bar(xpos - 0.19, vals, width=0.36, color=C["blue"], label="纯度 Tr ρ²")
    ax.bar(xpos + 0.19, ent, width=0.36, color=C["orange"], label="冯·诺依曼熵 S")
    for xx, v, e in zip(xpos, vals, ent):
        ax.text(xx - 0.19, v + 0.03, f"{v:.2f}", ha="center", fontsize=9, color=INK_2)
        ax.text(xx + 0.19, e + 0.03, f"{e:.2f}", ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(xpos); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.25)
    ax.set_title("纠缠的怪异之处：整体确定，部分完全随机")
    ax.text(0.5, -0.28, "整体是纯态（S=0，信息完备），但任一方单独看都是最大混合态（S=1，完全随机）——\n"
                        "信息不在 A 里，也不在 B 里，而在两者的关联中",
            transform=ax.transAxes, ha="center", fontsize=9, color=INK_2, linespacing=1.5)
    ax.legend(loc="upper right")
    return finish(fig, "06_bell_states_entropy.png",
                  "S(ρ_A) = 0 等价于可分离；S = 1 ebit = 一个「纠缠比特」，是量子隐形传态和密钥分发的燃料")


# --------------------------------------------------------------- CHSH
def measure_op(angle: float) -> torch.Tensor:
    """在 x-z 平面上沿 angle 方向的自旋测量算符 n·σ。"""
    return math.sin(angle) * q.X + math.cos(angle) * q.Z


def E_quantum(psi, a: float, b: float) -> float:
    op = torch.kron(measure_op(a), measure_op(b))
    v = q.vec(psi)
    return float(torch.vdot(v, op @ v).real)


def E_lhv(a: float, b: float, n: int = 200000, gen=None) -> float:
    """一个具体的局域隐变量模型：共享随机方向 λ，各自按 sign(cos(角度−λ)) 输出 ±1。"""
    lam = torch.rand(n, generator=gen, dtype=torch.float64) * 2 * math.pi
    A = torch.sign(torch.cos(torch.tensor(a) - lam))
    B = torch.sign(torch.cos(torch.tensor(b) - lam))
    return float((A * B).mean())


def sample_E(psi, a: float, b: float, shots: int, gen=None):
    """按量子概率抽样一次真实「实验」，返回关联 E 的估计与标准误。"""
    e = E_quantum(psi, a, b)
    # P(A=B) = (1+E)/2
    same = torch.rand(shots, generator=gen, dtype=torch.float64) < (1 + e) / 2
    vals = torch.where(same, 1.0, -1.0)
    return float(vals.mean()), float(vals.std() / math.sqrt(shots))


def fig_chsh(seed: int = 3):
    gen = torch.Generator().manual_seed(seed)
    psi = bell_states()["Φ⁺"]

    fig = plt.figure(figsize=(13.6, 8.0))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.24)

    # (a) 关联函数
    ax = fig.add_subplot(gs[0, 0])
    d = np.linspace(0, math.pi, 100)
    Eq = [E_quantum(psi, 0.0, -dd) for dd in d]
    El = [1 - 2 * dd / math.pi for dd in d]
    ax.plot(np.degrees(d), Eq, color=C["blue"], lw=2.4, label="量子力学 E = cos(a−b)")
    ax.plot(np.degrees(d), El, color=C["orange"], lw=2.2, ls="--",
            label="局域隐变量模型 E = 1 − 2Δ/π")
    samp_ang = np.linspace(0, math.pi, 13)
    pts = [E_lhv(0.0, -aa, 60000, gen) for aa in samp_ang]
    ax.plot(np.degrees(samp_ang), pts, "o", ms=5, color=C["orange"], alpha=0.85,
            label="隐变量模型蒙特卡洛")
    ax.fill_between(np.degrees(d), Eq, El, color=C["blue"], alpha=0.08)
    ax.set_xlabel("两个测量方向的夹角 Δ = |a − b| (度)")
    ax.set_ylabel("关联 E(a,b)")
    ax.set_title("① 量子关联比任何经典关联都「更相关」")
    ax.legend(loc="upper right", fontsize=8.5)

    # (b) CHSH 扫描
    ax = fig.add_subplot(gs[0, 1])
    ths = np.linspace(0, math.pi / 2, 200)
    S_q = []
    for t in ths:
        a, ap, b, bp = 0.0, 2 * t, t, 3 * t
        S_q.append(E_quantum(psi, a, b) - E_quantum(psi, a, bp)
                   + E_quantum(psi, ap, b) + E_quantum(psi, ap, bp))
    S_q = np.array(S_q)
    ax.plot(np.degrees(ths), S_q, color=C["blue"], lw=2.4, label="量子预言 S(θ) = 3cosθ − cos3θ")
    ax.axhline(2, color=C["red"], lw=2, ls="--", label="经典（定域实在论）上界 S ≤ 2")
    ax.axhline(2 * math.sqrt(2), color=C["violet"], lw=1.6, ls=":",
               label="西雷尔森上界 2√2 = 2.828")
    ax.fill_between(np.degrees(ths), 2, S_q, where=S_q > 2, color=C["blue"], alpha=0.12)
    imax = int(np.argmax(S_q))
    ax.plot([np.degrees(ths[imax])], [S_q[imax]], "o", ms=9, color=C["blue"],
            markeredgecolor="white", mew=1.6)
    ax.annotate(f"θ=45°, S = {S_q[imax]:.3f}", (np.degrees(ths[imax]), S_q[imax]),
                xytext=(56, 2.15), fontsize=9.5, color=C["blue"],
                arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1))
    ax.set_xlabel("角度参数 θ（a=0, a'=2θ, b=θ, b'=3θ）")
    ax.set_ylabel("CHSH 量 S")
    ax.set_ylim(0, 3.1)
    ax.set_title("② 贝尔-CHSH 不等式的量子破缺")
    ax.legend(loc="lower left", fontsize=8.2)
    print(f"  CHSH 最大值 S = {S_q[imax]:.4f}（理论 2√2 = {2*math.sqrt(2):.4f}）")

    # (c) 有限次实验：S 随测量次数的统计显著性
    ax = fig.add_subplot(gs[1, 0])
    t = math.pi / 4
    angles = [(0.0, t), (0.0, 3 * t), (2 * t, t), (2 * t, 3 * t)]
    signs = [1, -1, 1, 1]
    shots_list = [50, 100, 500, 1000, 5000, 20000, 100000]
    means, errs = [], []
    for shots in shots_list:
        tot, var = 0.0, 0.0
        for (a, b), sg in zip(angles, signs):
            m_, e_ = sample_E(psi, a, b, shots, gen)
            tot += sg * m_; var += e_ ** 2
        means.append(tot); errs.append(math.sqrt(var))
    means, errs = np.array(means), np.array(errs)
    ax.errorbar(shots_list, means, yerr=errs, fmt="o-", color=C["blue"], lw=2,
                ms=7, capsize=4, label="模拟实验 S ± 1σ")
    ax.axhline(2, color=C["red"], lw=2, ls="--", label="经典上界 2")
    ax.axhline(2 * math.sqrt(2), color=C["violet"], lw=1.4, ls=":", label="2√2")
    ax.set_xscale("log")
    ax.set_xlabel("每组角度的测量次数（对数轴）")
    ax.set_ylabel("实测 S")
    ax.set_title("③ 做实验：多少次测量能宣称「破缺」？")
    nsig = (means - 2) / errs
    for sh, mm, ee, ns in zip(shots_list, means, errs, nsig):
        if sh in (100, 5000, 100000):
            ax.annotate(f"{ns:.0f}σ", (sh, mm + ee + 0.06), ha="center",
                        fontsize=9, color=INK_2)
    ax.legend(loc="lower right", fontsize=8.5)
    print(f"  10 万次抽样：S = {means[-1]:.4f} ± {errs[-1]:.4f}，"
          f"超出经典上界 {nsig[-1]:.0f}σ")

    # (d) 不能传信
    ax = fig.add_subplot(gs[1, 1])
    bobs = np.linspace(0, 2 * math.pi, 60)
    pA_up, pAB = [], []
    for b in bobs:
        opA = torch.kron(measure_op(0.0), q.I2)
        v = q.vec(psi)
        eA = float(torch.vdot(v, opA @ v).real)
        pA_up.append((1 + eA) / 2)
        pAB.append((1 + E_quantum(psi, 0.0, b)) / 2)
    ax.plot(np.degrees(bobs), pA_up, color=C["blue"], lw=2.4,
            label="Alice 单独看到的 P(+1)：恒为 0.5")
    ax.plot(np.degrees(bobs), pAB, color=C["orange"], lw=2, ls="--",
            label="联合关联 P(A=B)：随 Bob 的角度变化")
    ax.set_ylim(-0.05, 1.38)
    ax.set_xlabel("Bob 选择的测量方向 b (度)")
    ax.set_ylabel("概率")
    ax.set_title("④ 纠缠为什么不能用来超光速通信")
    ax.text(0.03, 0.06,
            "Bob 怎么转他的仪器，Alice 那边的边缘分布都不动（蓝线全平）——\n"
            "关联只有在两人把结果拿到一起比对（经典信道，≤ c）之后才显现。",
            transform=ax.transAxes, fontsize=9, color=INK_2, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="#f4f4f2", ec=INK_MUTED, lw=0.8))
    ax.legend(loc="upper right", fontsize=8.5)
    print(f"  无信号定理检验：Alice 边缘概率极差 = {max(pA_up)-min(pA_up):.2e}")

    return finish(fig, "06_chsh_bell.png",
                  "贝尔定理：定域性 + 实在论 + 自由选择，三者必须放弃一个。实验站在量子力学一边（2022 诺贝尔物理学奖）")


def main():
    section("1935–2022 · 量子纠缠 —— 从 EPR 佯谬到贝尔实验")
    print(__doc__)
    fig_bell_and_entropy()
    fig_chsh()


if __name__ == "__main__":
    use_style()
    main()
