# CODE_INTEGRATION_NOTES.md

> 代码迁移备忘录（不是实验报告）。告诉以后的人：P10 里的代码和主项目代码怎么对应，
> 以及未来 Step 2 的新代码加在哪里。**防止把 P10 code 整包覆盖主项目。**

## 1. 当前原则

- 主项目代码以 `/share/home/yuan/SYX/eeg-mi-online/` 为准。
- P10 code（`/share/home/yuan/SYX/P10_MI泛化研究/code/`）**只读参考**。
- **不直接覆盖 `src/` `scripts/`**。

## 2. 已有主项目代码（baseline）

- `src/data/session_splits.py`
- `src/evaluation/session_protocols.py`
- `src/training/trainer.py`
- `src/models/{eegnet,deepconvnet,fbcnet,registry}.py`（+ `cap_eegnet.py` v1，未列入当前 baseline 运行）
- `scripts/train_session_models.py`
- `scripts/summarize_session_results.py`
- 漂移：`src/analysis/session_drift.py` + `scripts/analysis/run_session_drift.py`

## 3. Step 1 multi-source 已新增 / 恢复代码

- `src/evaluation/session_multisource_protocols.py`
- `scripts/train_session_multisource.py`
- `scripts/summarize_multisource_results.py`
- `configs/session_multisource_compare.yaml`
- `scripts/slurm/train_session_multisource_gpu.sbatch`
- `scripts/slurm/summarize_multisource_results_cpu.sbatch`

（这些在一次工具故障中从工作树丢失，已按运行版重新落盘；结果未受影响。复用现有 trainer / registry /
`load_ok_sessions` / `TrainSpec` / `_loader` / `_stratified_val` / `evaluate_predictions`。）

## 4. Step 2 未来建议新增代码（尚未创建）

- `src/adaptation/session_alignment.py`
- `src/adaptation/bn_adaptation.py`
- `src/evaluation/session_adaptation_protocols.py`
- `scripts/train_session_adaptation.py`
- `scripts/summarize_adaptation_results.py`
- `configs/session_adaptation_compare.yaml`
- `scripts/slurm/train_session_adaptation_gpu.sbatch`
- `scripts/slurm/summarize_adaptation_results_cpu.sbatch`

## 5. P10 code 对照

| P10 文件 | 归类 |
|---|---|
| `eegnet_cross_session.py` | 参考 baseline，但主项目已有更规范的 trainer / protocol |
| `session_drift_diagnostic.py` | 参考，主项目已有 `src/analysis/session_drift.py` |
| `multi_agent_encoder.py` / `multi_agent_pretrain.py` | **future 草稿**，不进入当前 Step 2 |
| `online_drift_monitor.py` / `online_drift_pipeline.py` | **future 草稿，未验证** |
| `drift_dashboard.py` / `subject_report_generator.py` | future 可参考（报告/可视化） |

## 6. 禁止事项

- 不复制粘贴覆盖主项目核心 trainer / protocol。
- 不引入新依赖（Riemannian 优先 numpy/scipy 自实现，不装 pyriemann）。
- 不修改 `workspace2` 原始数据。
- **不把 online 草稿写成已验证。**
