"""量子涌现 —— 「More is Different」：多，就是不一样

安德森 1972 年的名篇《More is Different》：把基本定律知道得再清楚，也推不出
凝聚态里的超导、磁性、超流。N 个粒子的希尔伯特空间维数是 2^N，
里面装着单个粒子身上根本不存在的结构。这就是「涌现」。

本模块用横场 Ising 模型（TFIM）作为最小样板：

    Ĥ = −J Σ_i Z_i Z_{i+1} − g Σ_i X_i

  · g/J → 0：Z 方向铁磁序（两个简并基态，Z₂ 对称性自发破缺）
  · g/J → ∞：所有自旋指向 X，顺磁无序
  · g/J = 1：量子临界点。能隙闭合、关联长度发散、纠缠熵出现对数标度
      S(l) = (c/6)·ln[(2L/π)sin(πl/L)] + const，  中心荷 c = 1/2（自由马约拉纳费米子）
    —— 一个连续的、有共形对称性的场论，从一串离散自旋里「涌现」了出来。

第二个涌现：经典世界本身。
  退相干（Zurek）：系统和环境纠缠后，对系统取偏迹，密度矩阵的非对角元
  以 |ρ₀₁(t)| = ∏_k |cos(2g_k t)| 衰减。环境自由度越多，衰减越快 ——
  这就是为什么猫不会又死又活：宏观物体在 10⁻²⁰ 秒量级就被环境「测量」了。
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import matplotlib.pyplot as plt

from .style import use_style, finish, section, C, SERIES, SEQ_BLUE, INK, INK_2, INK_MUTED

DT = torch.float64
Z_R = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=DT)
X_R = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=DT)
I_R = torch.eye(2, dtype=DT)


def _site(op, i, n):
    out = torch.ones(1, 1, dtype=DT)
    for k in range(n):
        out = torch.kron(out, op if k == i else I_R)
    return out


# 稀疏版本：2^N 维空间里稠密对角化会瞬间爆炸（N=12 就要几十秒/次），
# 而我们只需要最低的几个本征态 —— 用 Lanczos（scipy.sparse.linalg.eigsh）。
_SZ = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))
_SX = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
_SI = sp.identity(2, format="csr")


def _site_sp(op, i, n):
    out = sp.identity(1, format="csr")
    for k in range(n):
        out = sp.kron(out, op if k == i else _SI, format="csr")
    return out


def tfim_sparse(n: int, J: float = 1.0, g: float = 1.0) -> sp.csr_matrix:
    """开边界横场 Ising 的稀疏哈密顿量 H = −J Σ Z_i Z_{i+1} − g Σ X_i。"""
    Zs = [_site_sp(_SZ, i, n) for i in range(n)]
    Hm = sp.csr_matrix((2 ** n, 2 ** n))
    for i in range(n - 1):
        Hm = Hm - J * (Zs[i] @ Zs[i + 1])
    for i in range(n):
        Hm = Hm - g * _site_sp(_SX, i, n)
    return Hm.tocsr()


def lowest(n: int, J: float, g: float, k: int = 2):
    """返回最低 k 个本征值与基态波函数（torch 张量）。"""
    Hm = tfim_sparse(n, J, g)
    kk = min(k, 2 ** n - 2)
    ev, evec = spla.eigsh(Hm, k=kk, which="SA", tol=1e-10)
    order = np.argsort(ev)
    ev, evec = ev[order], evec[:, order]
    return ev, torch.tensor(evec[:, 0], dtype=DT)


def tfim_real(n: int, J: float = 1.0, g: float = 1.0) -> torch.Tensor:
    """稠密实对称版本（只在小尺寸做校验时用）。"""
    dim = 2 ** n
    Hm = torch.zeros((dim, dim), dtype=DT)
    Zs = [_site(Z_R, i, n) for i in range(n)]
    for i in range(n - 1):
        Hm -= J * (Zs[i] @ Zs[i + 1])
    for i in range(n):
        Hm -= g * _site(X_R, i, n)
    return Hm


def half_chain_entropy(psi: torch.Tensor, n: int, l: int, base: float = 2.0) -> float:
    """前 l 个格点与其余部分之间的冯·诺依曼熵。

    base=2 → 单位是 ebit；base=e → 单位是 nat（共形场论的标准约定）。
    对纯态只需做一次 SVD：施密特系数的平方就是约化密度矩阵的本征值。
    """
    M = psi.reshape(2 ** l, 2 ** (n - l))
    sv = torch.linalg.svdvals(M)
    p = (sv ** 2).clamp_min(1e-16)
    p = p / p.sum()
    return float(-(p * torch.log(p)).sum() / math.log(base))


# ------------------------------------------------------------ 图 1：量子相变
def fig_quantum_phase_transition():
    gs_vals = np.linspace(0.05, 2.0, 40)
    sizes = [6, 8, 10, 12]
    gaps, mags, ents, e_dens = {}, {}, {}, {}
    for n in sizes:
        gap_l, mag_l, ent_l, ed_l = [], [], [], []
        Zs = [_site_sp(_SZ, i, n) for i in range(n)]
        Mz = sum(Zs) / n
        Mz2 = (Mz @ Mz).toarray()
        for g in gs_vals:
            ev, psi0 = lowest(n, 1.0, float(g), k=2)
            gap_l.append(float(ev[1] - ev[0]))
            ed_l.append(float(ev[0]) / n)
            # 序参量用 √〈M_z²〉（有限尺寸下 〈M_z〉 恒为 0，因为 Z₂ 对称性未破缺）
            v = psi0.numpy()
            mag_l.append(math.sqrt(max(float(v @ (Mz2 @ v)), 0.0)))
            ent_l.append(half_chain_entropy(psi0, n, n // 2))
        gaps[n], mags[n], ents[n], e_dens[n] = gap_l, mag_l, ent_l, ed_l
        print(f"  N={n:2d} 完成：g=1 处能隙 = {np.interp(1.0, gs_vals, gap_l):.4f}，"
              f"半链纠缠熵 = {np.interp(1.0, gs_vals, ent_l):.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))
    ramp = [SEQ_BLUE[i] for i in (3, 6, 9, 12)]

    ax = axes[0, 0]
    for n, col in zip(sizes, ramp):
        ax.plot(gs_vals, mags[n], color=col, lw=2, label=f"N = {n}")
    ax.axvline(1.0, color=C["red"], ls="--", lw=1.6)
    ax.text(1.05, 0.85, "量子临界点 g/J = 1", color=C["red"], fontsize=9.5)
    ax.set_xlabel("横场强度 g / J"); ax.set_ylabel("序参量 √〈M_z²〉")
    ax.set_title("① 序参量：铁磁序在 g<J 一侧「涌现」")
    ax.legend(title="链长")

    ax = axes[0, 1]
    for n, col in zip(sizes, ramp):
        ax.plot(gs_vals, gaps[n], color=col, lw=2, label=f"N = {n}")
    ax.axvline(1.0, color=C["red"], ls="--", lw=1.6)
    ax.annotate("链越长，能隙在临界点\n压得越低 → 热力学极限下闭合",
                xy=(1.0, np.interp(1.0, gs_vals, gaps[12])), xytext=(0.10, 1.35),
                fontsize=9, color=INK_2,
                arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1))
    ax.set_xlabel("横场强度 g / J"); ax.set_ylabel("能隙 E₁ − E₀")
    ax.set_title("② 能隙闭合：相变的动力学指纹")
    ax.legend(title="链长")

    ax = axes[1, 0]
    for n, col in zip(sizes, ramp):
        ax.plot(gs_vals, ents[n], color=col, lw=2, label=f"N = {n}")
    ax.axvline(1.0, color=C["red"], ls="--", lw=1.6)
    ax.set_xlabel("横场强度 g / J"); ax.set_ylabel("半链纠缠熵 S (ebit)")
    ax.set_title("③ 只有临界点上，纠缠熵随链长持续增长")
    ax.set_ylim(0, 1.45)
    ax.annotate("有序相：S = 1 ebit\n（两重简并的「猫态」，与 N 无关）",
                xy=(0.35, 1.0), xytext=(0.10, 1.18), fontsize=8.6, color=INK_2,
                arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.9))
    ax.annotate("无序相：S → 0\n（近乎乘积态，与 N 无关）",
                xy=(1.75, 0.15), xytext=(1.30, 0.42), fontsize=8.6, color=INK_2,
                arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.9))
    ax.legend(title="链长", loc="upper right", fontsize=8.5, title_fontsize=8.5)

    # 内插图：临界点上 S ∝ (c/6)·ln N —— 对数发散，正是「涌现的场论」的签名
    ins = ax.inset_axes([0.42, 0.10, 0.34, 0.34])
    Sc = [np.interp(1.0, gs_vals, ents[n]) for n in sizes]
    lnN = np.log(np.array(sizes, dtype=float))
    k_, b_ = np.polyfit(lnN, Sc, 1)
    ins.plot(lnN, Sc, "o", ms=5, color=C["orange"])
    ins.plot(lnN, k_ * lnN + b_, color=C["blue"], lw=1.6)
    ins.set_title(f"g=1: S ≈ {k_:.3f}·lnN + b", fontsize=7.5, pad=3)
    ins.set_xlabel("ln N", fontsize=7); ins.set_ylabel("S", fontsize=7)
    ins.tick_params(labelsize=6.5)
    print(f"  临界点半链熵对链长的对数斜率 = {k_:.3f} ebit/lnN "
          f"→ c ≈ {6*k_*math.log(2):.2f}（理论 1/2）")

    ax = axes[1, 1]
    for n, col in zip(sizes, ramp):
        d2 = np.gradient(np.gradient(np.array(e_dens[n]), gs_vals), gs_vals)
        ax.plot(gs_vals[2:-2], d2[2:-2], color=col, lw=2, label=f"N = {n}")
    ax.axvline(1.0, color=C["red"], ls="--", lw=1.6)
    ax.set_xlabel("横场强度 g / J"); ax.set_ylabel("∂²(E₀/N)/∂g²")
    ax.set_title("④ 二阶量子相变：能量连续，二阶导趋于奇点")
    ax.text(0.03, 0.10,
            "热力学极限下这里是一个真正的发散点；\n"
            "有限链把它抹圆了，但链越长，极小值越深、越靠近 g=1。",
            transform=ax.transAxes, fontsize=8.8, color=INK_2, linespacing=1.5)
    ax.legend(title="链长", loc="lower right", fontsize=8.5, title_fontsize=8.5)

    fig.suptitle("横场 Ising 模型：从一串自旋里长出一个相变", fontsize=14,
                 fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    return finish(fig, "07_quantum_phase_transition.png",
                  "量子相变发生在 T=0，驱动它的不是热涨落而是量子涨落（不确定性原理强迫 Z 与 X 无法同时确定）")


# ------------------------------------------------------------ 图 2：纠缠熵标度
def fig_entanglement_scaling():
    """临界点上的纠缠熵标度：S(l) = (c/6)·ln[(2L/π)sin(πl/L)] + const（开边界）。

    注意共形场论的约定用自然对数，所以这里的熵一律以 nat 为单位。
    """
    n = 16
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8))

    results = {}
    for g, col, name in [(0.4, C["blue"], "g=0.4 有序相（面积律）"),
                         (1.0, C["orange"], "g=1.0 临界点（对数律）"),
                         (2.0, C["aqua"], "g=2.0 无序相（面积律）")]:
        _, psi0 = lowest(n, 1.0, g, k=2)
        ls = list(range(1, n))
        S = [half_chain_entropy(psi0, n, l, base=math.e) for l in ls]
        results[g] = (np.array(ls), np.array(S))
        ax.plot(ls, S, "o-", color=col, lw=2, ms=6, label=name)
    ax.set_xlabel(f"子系统大小 l（链长 L = {n}）")
    ax.set_ylabel("纠缠熵 S(l) (nat)")
    ax.set_title("① 面积律 vs 对数律")
    ax.text(0.03, 0.72, "远离临界点：S 与 l 无关\n（只取决于「切开几刀」= 面积律）",
            transform=ax.transAxes, fontsize=9, color=INK_2)
    ax.legend(loc="center", bbox_to_anchor=(0.55, 0.52), fontsize=8.5)

    def fit_c(L):
        _, p0 = lowest(L, 1.0, 1.0, k=2)
        lsL = np.arange(1, L)
        SL = np.array([half_chain_entropy(p0, L, int(l), base=math.e) for l in lsL])
        chordL = np.log(2 * L / math.pi * np.sin(math.pi * lsL / L))
        A = torch.tensor(np.stack([chordL, np.ones_like(chordL)], 1), dtype=DT)
        sol = torch.linalg.lstsq(A, torch.tensor(SL, dtype=DT).unsqueeze(1)).solution.squeeze()
        return chordL, SL, float(sol[0]), float(sol[1])

    chord, S, slope, intercept = fit_c(n)
    c_fit = 6 * slope
    trend = [(L, 6 * fit_c(L)[2]) for L in (8, 10, 12, 14, 16)]
    print("  中心荷拟合随链长的收敛： " +
          "，".join(f"L={L}: c={cc:.3f}" for L, cc in trend) + "（理论 1/2）")

    ax2.plot(chord, S, "o", ms=8, color=C["orange"], label=f"数值（L={n} 临界基态）")
    xx = np.linspace(chord.min(), chord.max(), 50)
    ax2.plot(xx, slope * xx + intercept, color=C["blue"], lw=2,
             label=f"最小二乘拟合：c = 6×斜率 = {c_fit:.3f}")
    ax2.plot(xx, (0.5 / 6) * (xx - chord.mean()) + S.mean(),
             color=INK_MUTED, lw=1.8, ls="--", label="理论斜率 c = 1/2")
    ax2.set_xlabel("ln[(2L/π)·sin(πl/L)]   （共形弦长）")
    ax2.set_ylabel("S(l) (nat)")
    ax2.set_title("② 临界点上「涌现」出共形对称性")
    ax2.legend(loc="upper left", fontsize=8.6)
    ax2.text(0.97, 0.06,
             "有限尺寸下 c 收敛得很慢：\n"
             + "  ".join(f"L={L}→{cc:.2f}" for L, cc in trend[1:])
             + "\n一串离散自旋在临界点上表现得像一个连续场论",
             transform=ax2.transAxes, ha="right", fontsize=8.6, color=INK_2, linespacing=1.6)
    return finish(fig, "07_entanglement_scaling.png",
                  "c=1/2 是伊辛普适类的中心荷：微观细节（晶格常数、耦合的具体形式）全被洗掉，只剩普适的临界行为——这就是涌现")


# ------------------------------------------------------------ 图 3：退相干
def fig_decoherence(seed: int = 11):
    gen = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 12, 1200, dtype=DT)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))

    ax = axes[0]
    for i, N in enumerate([1, 4, 16, 64, 256]):
        gk = 0.3 + 0.7 * torch.rand(N, generator=gen, dtype=DT)
        coh = torch.prod(torch.cos(2 * gk[None, :] * t[:, None]).abs(), dim=1)
        ax.plot(t, coh, color=SERIES[i], lw=2, label=f"环境自由度 N = {N}")
    ax.set_xlabel("时间 t"); ax.set_ylabel("相干性 |ρ₀₁(t)| / |ρ₀₁(0)|")
    ax.set_title("① 环境越大，叠加消失得越快")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8.5)

    ax = axes[1]
    Ns = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    tau = []
    for N in Ns:
        gk = 0.3 + 0.7 * torch.rand(int(N), generator=gen, dtype=DT)
        tt = torch.linspace(0, 12, 4000, dtype=DT)
        coh = torch.prod(torch.cos(2 * gk[None, :] * tt[:, None]).abs(), dim=1)
        idx = int((coh < math.exp(-1)).nonzero()[0]) if (coh < math.exp(-1)).any() else -1
        tau.append(float(tt[idx]))
    ax.loglog(Ns, tau, "o-", color=C["blue"], lw=2, ms=7, label="数值退相干时间 τ_D")
    ax.loglog(Ns, tau[0] * Ns ** -0.5, ls="--", color=C["orange"], lw=2,
              label="∝ N^(−1/2) 参考线")
    ax.set_xlabel("环境自由度数 N"); ax.set_ylabel("退相干时间 τ_D")
    ax.set_title("② 退相干时间随环境规模幂律下降")
    ax.legend(fontsize=8.5)
    ax.text(0.04, 0.06, "一只真实的猫 ~10²⁵ 个自由度\n→ τ_D ≈ 10⁻²⁰ s 量级",
            transform=ax.transAxes, fontsize=9, color=INK_2)

    # ③ 密度矩阵：叠加态 → 经典混合
    ax = axes[2]
    ts_show = [0.0, 0.35, 1.2]
    for j, tv in enumerate(ts_show):
        gk = 0.3 + 0.7 * torch.rand(20, generator=gen, dtype=DT)
        coh = float(torch.prod(torch.cos(2 * gk * tv).abs()))
        rho = np.array([[0.5, 0.5 * coh], [0.5 * coh, 0.5]])
        sub = ax.inset_axes([0.03 + j * 0.335, 0.34, 0.28, 0.46])
        sub.imshow(rho, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        for r in range(2):
            for c_ in range(2):
                sub.text(c_, r, f"{rho[r,c_]:.2f}", ha="center", va="center",
                         fontsize=9, color=INK)
        sub.set_xticks([]); sub.set_yticks([]); sub.grid(False)
        sub.set_title(f"t = {tv}\n相干度 {coh:.2f}", fontsize=9)
    ax.axis("off")
    ax.set_title("③ 叠加 → 经典混合：只有非对角元消失")
    ax.text(0.5, 0.20,
            "对角元（各结果的概率）一点没变，\n"
            "消失的只是非对角的相干项 —— 干涉能力没了，\n"
            "「又死又活」就退化成「非死即活，只是我们不知道」。",
            transform=ax.transAxes, ha="center", fontsize=9.2, color=INK_2, linespacing=1.6)
    return finish(fig, "07_decoherence.png",
                  "退相干解释了经典世界为何浮现，但它不解决「唯一结果」的测量问题——那仍是多世界/哥本哈根等诠释在争的东西")


def main():
    section("量子涌现 —— 多体系统、量子相变与经典世界的浮现")
    print(__doc__)
    fig_quantum_phase_transition()
    fig_entanglement_scaling()
    fig_decoherence()


if __name__ == "__main__":
    use_style()
    main()
