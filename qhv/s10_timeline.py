"""量子力学 125 年：时间轴与人物谱系

把前面九个模块串起来的一张地图。四个时代：
  旧量子论(1900–1924) → 量子力学建立(1925–1932) → 诠释之争与量子场论(1935–1979)
  → 量子信息时代(1980–今)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .style import use_style, finish, section, C, SERIES, INK, INK_2, INK_MUTED, SURFACE

ERAS = [
    ("旧量子论", 1898, 1924, C["blue"]),
    ("量子力学建立", 1925, 1934, C["orange"]),
    ("诠释之争与量子场论", 1935, 1979, C["aqua"]),
    ("量子信息时代", 1980, 2026, C["violet"]),
]

EVENTS = [
    (1900, "普朗克：能量子 E = hν\n黑体辐射公式"),
    (1905, "爱因斯坦：光量子\n解释光电效应"),
    (1913, "玻尔：定态与跃迁\n氢原子模型"),
    (1922, "施特恩-格拉赫实验\n空间量子化"),
    (1923, "康普顿散射\n光子有动量"),
    (1924, "德布罗意：物质波 λ=h/p"),
    (1925, "海森堡矩阵力学\n泡利不相容原理"),
    (1926, "薛定谔方程\n玻恩概率诠释"),
    (1927, "不确定性原理\n第五届索尔维会议\n电子衍射实验"),
    (1928, "狄拉克方程\n预言反物质"),
    (1932, "冯·诺依曼《数学基础》\n安德森发现正电子"),
    (1935, "EPR 佯谬\n薛定谔的猫"),
    (1947, "兰姆位移 → QED 重整化\n(施温格/费曼/朝永, 1948)"),
    (1957, "埃弗雷特：多世界诠释\nBCS 超导理论"),
    (1964, "贝尔不等式\n把哲学争论变成实验"),
    (1972, "Freedman-Clauser\n首次检验贝尔不等式"),
    (1982, "阿斯佩实验\n无克隆定理\n费曼提出量子模拟"),
    (1984, "BB84 量子密钥分发"),
    (1985, "多伊奇：通用量子计算机"),
    (1993, "量子隐形传态方案"),
    (1994, "肖尔算法：分解大数\n1995 量子纠错码"),
    (1996, "格罗弗搜索算法"),
    (1997, "首次隐形传态实验\n(因斯布鲁克)"),
    (2015, "无漏洞贝尔实验\n(Delft/NIST/维也纳)"),
    (2016, "墨子号量子科学实验卫星"),
    (2019, "谷歌宣称量子优越性"),
    (2020, "九章光量子计算原型机"),
    (2022, "诺贝尔物理学奖：\n阿斯佩、克劳泽、蔡林格"),
    (2024, "逻辑量子比特与纠错\n成为主战场"),
]

PEOPLE = [
    # 姓名, 生, 卒(None=在世), 关键工作年, 关键贡献, 诺奖年
    ("普朗克 Planck", 1858, 1947, 1900, "能量子假设", 1918),
    ("爱因斯坦 Einstein", 1879, 1955, 1905, "光量子 / 受激辐射 / EPR", 1921),
    ("玻恩 Born", 1882, 1970, 1926, "波函数的概率诠释", 1954),
    ("玻尔 Bohr", 1885, 1962, 1913, "定态跃迁 / 互补原理", 1922),
    ("薛定谔 Schrödinger", 1887, 1961, 1926, "波动方程 / 纠缠一词 / 猫", 1933),
    ("德布罗意 de Broglie", 1892, 1987, 1924, "物质波 λ = h/p", 1929),
    ("泡利 Pauli", 1900, 1958, 1925, "不相容原理 / 自旋矩阵", 1945),
    ("海森堡 Heisenberg", 1901, 1976, 1925, "矩阵力学 / 不确定性原理", 1932),
    ("狄拉克 Dirac", 1902, 1984, 1928, "相对论性波动方程 / 反物质", 1933),
    ("冯·诺依曼 von Neumann", 1903, 1957, 1932, "量子力学的数学基础 / 测量", None),
    ("费曼 Feynman", 1918, 1988, 1948, "路径积分 / QED / 量子计算构想", 1965),
    ("贝尔 Bell", 1928, 1990, 1964, "贝尔不等式", None),
    ("克劳泽 Clauser", 1942, None, 1972, "首次贝尔实验", 2022),
    ("蔡林格 Zeilinger", 1945, None, 1997, "隐形传态实验 / 纠缠交换", 2022),
    ("阿斯佩 Aspect", 1947, None, 1982, "快速切换的贝尔实验", 2022),
    ("肖尔 Shor", 1959, None, 1994, "大数分解算法 / 纠错码", None),
    ("潘建伟 Pan", 1970, None, 2016, "墨子号 / 洲际量子密钥分发", None),
]


def fig_timeline():
    fig, ax = plt.subplots(figsize=(17.5, 8.4))
    ax.set_xlim(1896, 2030)
    ax.set_ylim(-5.6, 5.6)
    ax.axis("off")

    for name, a, b, col in ERAS:
        ax.add_patch(plt.Rectangle((a, -0.30), b - a, 0.60, fc=col, alpha=0.18, lw=0))
    ax.plot([1896, 2030], [0, 0], color=INK_MUTED, lw=1.2, zorder=0)

    # 时代图例单独放一行，避免和事件连线抢地方
    for i, (name, a, b, col) in enumerate(ERAS):
        x0 = 1899 + i * 33
        ax.add_patch(plt.Rectangle((x0, 5.15), 3.2, 0.30, fc=col, alpha=0.55, lw=0))
        ax.text(x0 + 4.2, 5.30, f"{name}  {a if a > 1898 else 1900}–{b if b < 2026 else '今'}",
                ha="left", va="center", fontsize=10, color=col, fontweight="bold")

    def era_color(year):
        for _, a, b, col in ERAS:
            if a <= year <= b:
                return col
        return INK_MUTED

    levels_up = [1.15, 2.35, 3.55, 4.75]
    levels_dn = [-1.15, -2.35, -3.55, -4.75]
    up_last, dn_last = -999, -999
    up_i, dn_i = 0, 0
    for k, (year, text) in enumerate(EVENTS):
        col = era_color(year)
        if k % 2 == 0:
            up_i = (up_i + 1) % len(levels_up) if year - up_last < 12 else 0
            y = levels_up[up_i]; up_last = year
            va = "bottom"
        else:
            dn_i = (dn_i + 1) % len(levels_dn) if year - dn_last < 12 else 0
            y = levels_dn[dn_i]; dn_last = year
            va = "top"
        ax.plot([year, year], [0, y * 0.86], color=col, lw=1.1, alpha=0.75, zorder=1)
        ax.plot([year], [0], "o", ms=7, color=col, markeredgecolor="white",
                markeredgewidth=1.4, zorder=3)
        ax.text(year, y * 0.86 + (0.06 if va == "bottom" else -0.06),
                f"{year}\n{text}", ha="center", va=va, fontsize=8.4,
                color=INK, linespacing=1.45,
                bbox=dict(boxstyle="round,pad=0.34", fc=SURFACE, ec=col, lw=0.9, alpha=0.96))

    for yr in range(1900, 2031, 10):
        ax.text(yr, -0.55, str(yr), ha="center", va="top", fontsize=8, color=INK_MUTED)
    ax.set_title("量子力学 125 年：从一次「绝望的凑数」到一个产业", fontsize=16,
                 fontweight="bold", pad=16)
    return finish(fig, "10_timeline.png",
                  "上半轴与下半轴交替排布，节点颜色标示所属时代；每个事件都对应本项目中的一段可运行代码")


def fig_people():
    fig, ax = plt.subplots(figsize=(14.6, 8.6))
    order = sorted(PEOPLE, key=lambda p: p[1])
    ys = np.arange(len(order))[::-1]

    for (name, born, died, key_year, contrib, nobel), y in zip(order, ys):
        end = died if died else 2026
        alive_now = died is None
        ax.plot([born, end], [y, y], color=INK_MUTED, lw=6, alpha=0.22,
                solid_capstyle="round")
        ax.plot([key_year, end], [y, y], color=C["blue"], lw=6, alpha=0.0)
        ax.plot([key_year], [y], "o", ms=13, color=C["blue"],
                markeredgecolor="white", markeredgewidth=1.8, zorder=4)
        ax.text(key_year - 2.5, y, f"{key_year}", ha="right", va="center",
                fontsize=8.5, color=C["blue"], fontweight="bold")
        ax.text(key_year + 3.0, y + 0.30, contrib, ha="left", va="center",
                fontsize=9, color=INK)
        if nobel:
            ax.plot([nobel], [y], "*", ms=15, color=C["yellow"],
                    markeredgecolor="white", markeredgewidth=0.9, zorder=5)
        ax.text(born - 2.5, y, f"{name}", ha="right", va="center", fontsize=9.5,
                color=INK, fontweight="bold")
        life = f"{born}–{died if died else '   '}"
        ax.text(born - 2.5, y - 0.34, life, ha="right", va="center", fontsize=7.5,
                color=INK_MUTED)
        if alive_now:
            ax.plot([end], [y], ">", ms=7, color=INK_MUTED, alpha=0.5)

    ax.set_xlim(1820, 2062)
    ax.set_ylim(-1.2, len(order) - 0.2)
    ax.set_yticks([])
    ax.set_xticks(range(1850, 2051, 25))
    ax.set_xlabel("年份")
    ax.grid(True, axis="x", alpha=0.4)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)

    ax.plot([], [], color=INK_MUTED, lw=6, alpha=0.22, label="生卒年")
    ax.plot([], [], "o", ms=11, color=C["blue"], label="关键工作发表年")
    ax.plot([], [], "*", ms=14, color=C["yellow"], label="诺贝尔物理学奖")
    ax.legend(loc="lower right", fontsize=9.5, ncol=3)
    ax.set_title("量子力学的缔造者们：生命线、关键工作与诺奖", fontsize=15,
                 fontweight="bold", pad=14)
    ax.text(1822, -0.95,
            "1927 年第五届索尔维会议合影里的 29 个人，有 17 位后来拿了诺贝尔奖 —— "
            "量子力学是人类历史上智力最密集的一次集体攻关。",
            fontsize=9, color=INK_2)
    return finish(fig, "10_people.png",
                  "注意年龄：海森堡提出矩阵力学时 24 岁，狄拉克写下狄拉克方程时 26 岁，泡利提出不相容原理时 25 岁——所以那时人们叫它「男孩物理学」")


def main():
    section("量子力学 125 年 —— 时间轴与人物谱系")
    print(__doc__)
    fig_timeline()
    fig_people()


if __name__ == "__main__":
    use_style()
    main()
