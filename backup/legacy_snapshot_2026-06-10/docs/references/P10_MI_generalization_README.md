# P10_MI_generalization_README.md

> 外部 P10 文件夹的索引（导航页，不是实验报告）。整合解读见 `docs/P10_INTEGRATION_SUMMARY.md`。

## 1. 外部路径

```
/share/home/yuan/SYX/P10_MI泛化研究/
```

## 2. 用途

- 学长方向调整后的**参考包**。
- **只读参考**。
- **不作为主项目运行目录**（主项目是 `/share/home/yuan/SYX/eeg-mi-online/`）。

## 3. 主要文件

| 文件/目录 | 内容 |
|---|---|
| `README.md` | P10 总览 |
| `NEXT_STEPS.md` | 下一步计划 |
| `GPT_HANDOFF.md` | 给 GPT 的交接提示 |
| `proposal.md` | 研究提案 |
| `experiment_log.md` | 实验记录 |
| `session_drift/` | Phase 0 漂移结果 |
| `result_summary/` | Phase 1 baseline 结果 |
| `code/` | Phase 0-2 草稿代码 |

## 4. 整合结论

- **Phase 0/1 与主项目结果一致**（同一研究）。
- **Phase 2 online code 未验证**（草稿/设计）。
- 当前**只采纳方向，不直接运行 P10 code**。

## 5. 对应主项目路径

| P10 内容 | 主项目对应 |
|---|---|
| 整合说明 | `docs/P10_INTEGRATION_SUMMARY.md` |
| 代码对照 | `docs/CODE_INTEGRATION_NOTES.md` |
| Phase 0 drift | `outputs/analysis/session_drift_v1/` |
| Phase 1 baseline | `outputs/experiments/baseline_v1/provenance/session_model_compare_v1/` |
| (Step 1 多源，主项目新增) | `outputs/experiments/baseline_v1/provenance/session_multisource_v1/` |

> 它的作用就是"索引"，不是实验报告。
