# references/README.md

> 参考材料总索引（导航页）。`references/` 不是运行入口；不要把参考代码当正式代码。

## 1. 参考材料分类

- **数据集论文**：WBCIC-SHU 2C（Scientific Data，`s41597-025-04826-y`）。PDF gitignored，按需放入。
- **学长 P10 文件夹**：方向来源（外部，见 §2）。
- **senior scripts**：学长原始脚本（`senior_scripts/`，只读，已在主项目重实现）。
- **ChatGPT 设计文档**：`ChatGPT-EEG-MI-pretraining.md`（长期 CAP-EEGNet 愿景来源）。
- **PPT / PDF**：数据集汇报 slides、论文（体积大，不进 git）。

## 2. P10 外部引用

```
/share/home/yuan/SYX/P10_MI泛化研究/
```

- **只读，不直接运行。** 文件清单见 `docs/references/P10_MI_generalization_README.md`；
  整合解读见 `docs/P10_INTEGRATION_SUMMARY.md`。
- 早期本地快照：`docs/references/P10_MI_generalization/`。

## 3. 主项目内对应实现

| 任务 | 主项目实现 |
|---|---|
| session_drift | `src/analysis/session_drift.py`（+ `scripts/analysis/run_session_drift.py`） |
| baseline（within + single-source cross） | `scripts/train_session_models.py`（+ `src/evaluation/session_protocols.py`） |
| multisource（ses-01+02→ses-03） | `scripts/train_session_multisource.py`（+ `src/evaluation/session_multisource_protocols.py`） |
| adaptation（Step 2） | **尚未实现**，见 `docs/ADAPTATION_BASELINE_PLAN.md` |
| online | **future work**（P10 `code/` 是未验证草稿） |

## 4. 注意

- `references/` **不是运行入口**。
- **不要把参考代码当正式代码**；运行用 `src/` + `scripts/` 的规范版本。
- senior scripts ↔ 主项目实现的逐项映射：
  - `senior_scripts/data_validation/session_drift_diagnostic.py` → `src/analysis/session_drift.py`
  - `senior_scripts/model_training/eegnet_cross_session.py` → `src/models/*` + `src/training/trainer.py` + `src/evaluation/session_protocols.py`
