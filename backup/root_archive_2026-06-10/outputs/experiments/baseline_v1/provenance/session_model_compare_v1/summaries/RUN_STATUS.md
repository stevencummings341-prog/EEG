# Overnight baseline run status — 2026-06-07T21:39:01

**Overall: INCOMPLETE / NEEDS ATTENTION**

## Training jobs (sacct)

| job_id | protocol | seed | state | elapsed | exit |
|---|---|---|---|---|---|
| 21161 | within | 0 | COMPLETED | 09:24:11 | 0:0 |
| 21162 | cross | 0 | COMPLETED | 01:46:43 | 0:0 |
| 21163 | within | 1 | COMPLETED | 09:14:26 | 0:0 |
| 21164 | cross | 1 | COMPLETED | 01:46:52 | 0:0 |
| 21165 | within | 2 | COMPLETED | 09:21:34 | 0:0 |
| 21166 | cross | 2 | COMPLETED | 02:22:33 | 0:0 |
| 21167 | within | 3 | COMPLETED | 11:21:41 | 0:0 |
| 21168 | cross | 3 | COMPLETED | 02:26:16 | 0:0 |
| 21169 | within | 4 | COMPLETED | 10:48:18 | 0:0 |
| 21170 | cross | 4 | COMPLETED | 02:16:08 | 0:0 |
| 21171 | report | afterany | RUNNING | 00:00:02 | 0:0 |

Non-COMPLETED jobs: ['21171'].

## Summarizer

- ran OK: **True**
```
[summarize] 26520 rows | within=22200 cross=4320 | models=['deepconvnet', 'eegnet', 'fbcnet'] | seeds=[np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)]
[summarize] wrote tables + 3 figures + report to /share/home/yuan/SYX/eeg-mi-online/outputs/experiments/session_model_compare_v1/summaries | incomplete=False
```

## Expected output files

| file | present |
|---|---|
| `results_within_session.csv` | yes |
| `results_cross_session.csv` | yes |
| `within_by_seed.csv` | yes |
| `cross_by_seed.csv` | yes |
| `within_session_wise.csv` | yes |
| `cross_by_direction.csv` | yes |
| `summary_by_model_protocol.csv` | yes |
| `model_ranking.md` | yes |
| `SESSION_MODEL_COMPARE_REPORT.md` | yes |
| `within_session_accuracy_boxplot.png` | yes |
| `cross_session_accuracy_matrix_by_model.png` | yes |
| `protocol_comparison.png` | yes |

See `SESSION_MODEL_COMPARE_REPORT.md` for the full metrics report.
