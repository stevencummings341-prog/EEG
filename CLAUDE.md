---
title: "CLAUDE.md — 兼容入口（指向 AGENTS.md）"
tags:
  - "#modality/eeg"
created: "2026-06-12"
updated: "2026-07-06"
status: "active"
---

# CLAUDE.md — 兼容入口

> 本文件只是兼容入口，**唯一权威项目记忆是同目录的 `AGENTS.md`**。
> Claude Code / 其他会自动读取 `CLAUDE.md` 的工具，请立刻转读下列文件再开工。

## 接手前必读（按顺序）

1. **`AGENTS.md`** — 唯一权威「灵魂记忆」：项目身份、研究链、已验证事实、规则、硬边界。
2. `0_docs/ARCHITECTURE.md` — 目录结构 + 代码分层 + 文件职责。
3. `0_docs/STATUS.md` — 当前进度、能否跑、SHU 就绪度、下一步、清理策略。
4. `progress.md` — 进度日记（逐条追加，最新在上）。
5. `0_docs/FILE_CATALOG.md` — 文件索引（新增文件后回来更新）。

> 核心灵魂集（导入 ChatGPT 做背景更新）= `AGENTS.md` + `progress.md` + `0_docs/STATUS.md` + `0_docs/ARCHITECTURE.md`。

## 最关键的三条（细节见 AGENTS.md）

- 工作根目录固定 `/share/home/yuan/SYX/eeg-mi-online`；唯一人工入口 `python code/run.py --config code/configs/experiments/<phase>.yaml`。
- 外部 raw 数据 `/share/workspace2/moto_imagination/{WBCIC_SHU,SHU}` 只读；不编造结果；GPU 走 Slurm + `mi_torch_cu118`，禁登录节点跑重活。
- 每做一件事立刻同步文档（`progress.md` / `0_docs/operation_log.md` / `0_docs/FILE_CATALOG.md` / 最近 README），知识不许只留在聊天里。

注意：`/share/home/yuan/SYX` 下现在只有本项目 `eeg-mi-online/`（+ `backups/` + 无关的 `run_test.sh`）；旧的根目录 `CLAUDE.md` 与 `P10_MI泛化研究/` 已删除。本项目唯一权威是本目录的 `AGENTS.md`。
