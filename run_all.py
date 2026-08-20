#!/usr/bin/env python3
"""一键跑完全部章节，把 27 张图输出到 figures/。

    python run_all.py            # 全部
    python run_all.py 6 8        # 只跑第 6、8 章
"""
from __future__ import annotations

import importlib
import sys
import time

from qhv.style import use_style

CHAPTERS = [
    ("01", "qhv.s01_planck", "普朗克：能量子与黑体辐射"),
    ("02", "qhv.s02_einstein_bohr", "爱因斯坦 / 玻尔 / 德布罗意"),
    ("03", "qhv.s03_schrodinger", "薛定谔方程与玻恩诠释"),
    ("04", "qhv.s04_heisenberg", "海森堡：矩阵力学与不确定性"),
    ("05", "qhv.s05_dirac", "狄拉克：反物质与颤动"),
    ("06", "qhv.s06_entanglement", "量子纠缠与贝尔不等式"),
    ("07", "qhv.s07_emergence", "量子涌现：相变与经典世界"),
    ("08", "qhv.s08_computing", "量子计算：Grover / Shor / VQE"),
    ("09", "qhv.s09_communication", "量子通信：隐形传态与 BB84"),
    ("10", "qhv.s10_timeline", "125 年时间轴与人物谱系"),
]


def main(argv: list[str]) -> None:
    use_style()
    wanted = {a.zfill(2) for a in argv} if argv else None
    t0 = time.time()
    for num, mod_name, title in CHAPTERS:
        if wanted and num not in wanted:
            continue
        mod = importlib.import_module(mod_name)
        mod.main()
    print(f"\n全部完成，用时 {time.time() - t0:.1f}s。图片在 figures/ 目录下。")


if __name__ == "__main__":
    main(sys.argv[1:])
