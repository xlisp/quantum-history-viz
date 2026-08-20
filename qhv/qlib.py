"""qlib —— 一个用 PyTorch 从零写的极简量子力学 / 量子计算工具库。

设计要点：
  * 态矢量 psi 用形状 (2,)*n 的 complex128 张量表示，第 k 个轴 = 第 k 个量子比特；
    这样施加门 = 对某几个轴做张量收缩，天然支持自动微分（VQE 就靠它）。
  * 所有函数都是纯函数，不改原张量，方便 autograd 反向传播。
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch

CDTYPE = torch.complex128
RDTYPE = torch.float64

# ---------------------------------------------------------------- 基本矩阵
def _c(m) -> torch.Tensor:
    return torch.tensor(m, dtype=CDTYPE)

I2 = _c([[1, 0], [0, 1]])
X = _c([[0, 1], [1, 0]])
Y = _c([[0, -1j], [1j, 0]])
Z = _c([[1, 0], [0, -1]])
H = _c([[1, 1], [1, -1]]) / math.sqrt(2)
S = _c([[1, 0], [0, 1j]])
T = _c([[1, 0], [0, math.cos(math.pi / 4) + 1j * math.sin(math.pi / 4)]])
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def rx(theta) -> torch.Tensor:
    th = torch.as_tensor(theta, dtype=RDTYPE)
    c, s = torch.cos(th / 2), torch.sin(th / 2)
    return torch.stack([torch.stack([c + 0j, -1j * s]),
                        torch.stack([-1j * s, c + 0j])]).to(CDTYPE)


def ry(theta) -> torch.Tensor:
    th = torch.as_tensor(theta, dtype=RDTYPE)
    c, s = torch.cos(th / 2), torch.sin(th / 2)
    return torch.stack([torch.stack([c + 0j, -s + 0j]),
                        torch.stack([s + 0j, c + 0j])]).to(CDTYPE)


def rz(theta) -> torch.Tensor:
    th = torch.as_tensor(theta, dtype=RDTYPE)
    e = torch.exp(-0.5j * th.to(CDTYPE))
    return torch.stack([torch.stack([e, torch.zeros((), dtype=CDTYPE)]),
                        torch.stack([torch.zeros((), dtype=CDTYPE), e.conj()])])


CNOT = _c([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CZ = _c([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
SWAP = _c([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


# ---------------------------------------------------------------- 态与门
def zero_state(n: int) -> torch.Tensor:
    """|00...0>"""
    psi = torch.zeros((2,) * n, dtype=CDTYPE)
    psi[(0,) * n] = 1.0
    return psi


def basis_state(bits: Sequence[int]) -> torch.Tensor:
    psi = torch.zeros((2,) * len(bits), dtype=CDTYPE)
    psi[tuple(bits)] = 1.0
    return psi


def apply1(psi: torch.Tensor, U: torch.Tensor, q: int) -> torch.Tensor:
    """把单比特门 U 作用到第 q 个比特上。"""
    out = torch.tensordot(U.to(CDTYPE), psi, dims=([1], [q]))
    return torch.movedim(out, 0, q)


def apply2(psi: torch.Tensor, U: torch.Tensor, q0: int, q1: int) -> torch.Tensor:
    """把两比特门 U(4x4) 作用到 (q0, q1) 上，q0 为高位（控制位在前）。"""
    U4 = U.to(CDTYPE).reshape(2, 2, 2, 2)          # (o0,o1,i0,i1)
    out = torch.tensordot(U4, psi, dims=([2, 3], [q0, q1]))
    return torch.movedim(out, (0, 1), (q0, q1))


def cnot(psi, control: int, target: int) -> torch.Tensor:
    return apply2(psi, CNOT, control, target)


def vec(psi: torch.Tensor) -> torch.Tensor:
    """展平成 2^n 维列向量（比特 0 为最高位，即大端序）。"""
    return psi.reshape(-1)


def probs(psi: torch.Tensor) -> torch.Tensor:
    return (vec(psi).abs() ** 2).to(RDTYPE)


def sample(psi: torch.Tensor, shots: int, generator=None) -> torch.Tensor:
    p = probs(psi)
    return torch.multinomial(p / p.sum(), shots, replacement=True, generator=generator)


def bitstrings(n: int) -> list[str]:
    return [format(i, f"0{n}b") for i in range(2 ** n)]


# ---------------------------------------------------------------- 密度矩阵
def density(psi: torch.Tensor) -> torch.Tensor:
    v = vec(psi)
    return torch.outer(v, v.conj())


def partial_trace_pure(psi: torch.Tensor, keep: Iterable[int]) -> torch.Tensor:
    """对纯态求约化密度矩阵 rho_keep = Tr_rest |psi><psi|。"""
    n = psi.dim()
    keep = list(keep)
    rest = [q for q in range(n) if q not in keep]
    M = psi.permute(*keep, *rest).reshape(2 ** len(keep), 2 ** len(rest))
    return M @ M.conj().T


def partial_trace(rho: torch.Tensor, n: int, keep: Iterable[int]) -> torch.Tensor:
    """对一般密度矩阵求偏迹。"""
    keep = list(keep)
    rest = [q for q in range(n) if q not in keep]
    t = rho.reshape((2,) * (2 * n))
    perm = keep + rest + [q + n for q in keep] + [q + n for q in rest]
    t = t.permute(*perm).reshape(2 ** len(keep), 2 ** len(rest),
                                 2 ** len(keep), 2 ** len(rest))
    return torch.einsum("aibi->ab", t)


def entropy_vn(rho: torch.Tensor, base: float = 2.0) -> torch.Tensor:
    """冯·诺依曼熵 S = -Tr rho log rho（默认以 2 为底，单位 = ebit）。"""
    ev = torch.linalg.eigvalsh(rho).real.clamp_min(1e-15)
    return float(-(ev * torch.log(ev) / math.log(base)).sum())


def purity(rho: torch.Tensor) -> float:
    return float((rho @ rho).diagonal().sum().real)


def fidelity_pure(psi: torch.Tensor, phi: torch.Tensor) -> float:
    return float((torch.vdot(vec(psi), vec(phi)).abs() ** 2).real)


def bloch_vector(rho: torch.Tensor) -> tuple[float, float, float]:
    return tuple(float((rho @ P).diagonal().sum().real) for P in (X, Y, Z))


# ---------------------------------------------------------------- 多体算符
def kron_list(ops: Sequence[torch.Tensor]) -> torch.Tensor:
    out = ops[0]
    for op in ops[1:]:
        out = torch.kron(out, op)
    return out


def site_op(op: torch.Tensor, site: int, n: int) -> torch.Tensor:
    """把单点算符放到 n 体希尔伯特空间的第 site 个位置。"""
    return kron_list([op if k == site else I2 for k in range(n)])


def expect(op: torch.Tensor, psi_or_rho: torch.Tensor) -> float:
    a = psi_or_rho
    if a.dim() == 1 or (a.dim() > 1 and a.shape[0] != a.shape[-1]):
        v = a.reshape(-1)
        return float((torch.vdot(v, op @ v)).real)
    if a.dim() != 2:
        v = a.reshape(-1)
        return float((torch.vdot(v, op @ v)).real)
    return float((op @ a).diagonal().sum().real)


def tfim_hamiltonian(n: int, J: float = 1.0, g: float = 1.0,
                     periodic: bool = False) -> torch.Tensor:
    """横场 Ising 模型  H = -J Σ Z_i Z_{i+1} - g Σ X_i"""
    dim = 2 ** n
    Hm = torch.zeros((dim, dim), dtype=CDTYPE)
    bonds = range(n) if periodic else range(n - 1)
    for i in bonds:
        Hm = Hm - J * (site_op(Z, i, n) @ site_op(Z, (i + 1) % n, n))
    for i in range(n):
        Hm = Hm - g * site_op(X, i, n)
    return Hm


def ground_state(Hm: torch.Tensor) -> tuple[float, torch.Tensor]:
    ev, evec = torch.linalg.eigh(Hm)
    return float(ev[0].real), evec[:, 0]
