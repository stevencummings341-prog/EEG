#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scaffold per-level README.md across the two-dataset result tree.

用途: 为 1_session_drift / 2_baseline / 3_online_adaptation / 4_experiments /
5_papers 下的「数据集并列」结构补齐每一层的 README.md（数据集层、实验层、叶子层）。
顶层 README 由人工维护，本脚本默认不覆盖顶层(只在缺失时建占位)。

输入: 无（按下方 SPEC 静态生成）。
输出: 各级目录的 README.md（已存在则跳过，除非 --force）。
依赖: 仅标准库。

运行:
  python scripts/scaffold_readmes.py            # 只补缺失
  python scripts/scaffold_readmes.py --force    # 覆盖叶子/实验/数据集层 README
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()

DATASETS = {
    "wbci_shu": "WBCIC-SHU 2025 (62 人 × 3 天 × 58ch, 2C 左右手抓握)",
    "shu": "SHU 2022 (25 人 × 5 天 × 32ch, 2C 左右手抓握)",
}

# phase 目录 -> (中文名, 各数据集下的实验子目录列表, 叶子目录列表)
PHASES = {
    "1_session_drift": {
        "title": "Phase 0 跨 Session 漂移诊断",
        "experiments": {"": "漂移诊断 (MMD / CSP / ERD / 信号质量)"},
        "leaves": ["report", "tables", "figures"],
    },
    "2_baseline": {
        "title": "Phase 1/2a/2b Baseline 与 Alignment",
        "experiments": {
            "no_alignment_baseline": "Phase 1 within/cross + Phase 2a multi-source (无对齐)",
            "alignment_baseline": "Phase 2b no-learning alignment (有对齐)",
        },
        "leaves": ["report", "tables", "figures"],
    },
    "3_online_adaptation": {
        "title": "Phase 2a 在线适应框架",
        "experiments": {"": "在线 test-then-update 框架设计与结果"},
        "leaves": [],
    },
    "4_experiments": {
        "title": "Phase 2c+ 新实验",
        "experiments": {"prototype_drift": "Prototype Drift Analysis (嵌入空间原型漂移诊断)"},
        "leaves": ["report", "tables", "figures"],
    },
    "5_papers": {
        "title": "论文产出",
        "experiments": {"": "论文草稿 / 图表 / 投稿材料"},
        "leaves": [],
    },
}

LEAF_DESC = {
    "report": "文字报告 (`*_REPORT.md`)、配置与来源快照。",
    "tables": "结果数据表 (`*.csv` / `*.json`)，列名英文小写下划线。",
    "figures": "图表 (`*.png`)。",
}


def frontmatter(title: str) -> str:
    return (
        "---\n"
        f'title: "{title}"\n'
        "tags:\n"
        '  - "#modality/eeg"\n'
        '  - "#method/domain_generalization"\n'
        f'created: "{TODAY}"\n'
        f'updated: "{TODAY}"\n'
        'status: "active"\n'
        "---\n\n"
    )


def write(path: Path, body: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        # 顶层 README 永不覆盖；其余仅在缺失时写
        return
    path.write_text(body, encoding="utf-8")
    print(("OVERWRITE" if path.exists() else "CREATE"), path.relative_to(PROJECT_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="覆盖数据集/实验/叶子层 README")
    args = ap.parse_args()

    for phase, spec in PHASES.items():
        phase_dir = PROJECT_ROOT / phase
        for ds, ds_desc in DATASETS.items():
            ds_dir = phase_dir / ds
            # 数据集层 README
            exp_lines = "\n".join(
                f"- `{name or '(本层)'}/`: {desc}" for name, desc in spec["experiments"].items()
            )
            write(
                ds_dir / "README.md",
                frontmatter(f"{phase} · {ds} · {spec['title']}")
                + f"# {phase} — {ds}\n\n"
                f"数据集: {ds_desc}\n\n"
                f"本目录是 **{spec['title']}** 在 {ds} 上的结果区。\n\n"
                "## 实验\n\n" + (exp_lines or "- (待补)") + "\n\n"
                "## 约定\n\n"
                "- 每个实验下按 `report/ tables/ figures/` 分层。\n"
                "- 已完成结果只读；复跑用新 `run_id`，不覆盖历史。\n"
                "- 命名规范见 `AGENTS.md` / `CLAUDE.md` 第 9 节。\n",
                args.force,
            )
            # 实验层 + 叶子层
            for exp_name, exp_desc in spec["experiments"].items():
                exp_dir = ds_dir / exp_name if exp_name else ds_dir
                if exp_name:
                    write(
                        exp_dir / "README.md",
                        frontmatter(f"{ds} · {exp_name}")
                        + f"# {exp_name} — {ds}\n\n{exp_desc}\n\n"
                        "## 结构\n\n"
                        + "".join(f"- `{lf}/`: {LEAF_DESC[lf]}\n" for lf in spec["leaves"])
                        + "\n已完成结果只读;复跑用新 `run_id`。\n",
                        args.force,
                    )
                for leaf in spec["leaves"]:
                    write(
                        exp_dir / leaf / "README.md",
                        frontmatter(f"{ds} · {exp_name or phase} · {leaf}")
                        + f"# {leaf} — {ds} / {exp_name or spec['title']}\n\n"
                        f"{LEAF_DESC[leaf]}\n\n"
                        "空目录请保留 `.gitkeep`;有产出后删除占位说明即可。\n",
                        args.force,
                    )


if __name__ == "__main__":
    main()
