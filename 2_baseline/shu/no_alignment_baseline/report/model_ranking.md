# Model ranking — within vs cross-session (baselines, 5-seed)

**Ranked by cross-session accuracy:**

1. `eegnet` — cross 0.5381, within 0.6113, drop 0.0732 (12.0%)
2. `deepconvnet` — cross 0.5363, within 0.6064, drop 0.0701 (11.6%)
3. `fbcnet` — cross 0.5078, within 0.5535, drop 0.0457 (8.2%)

**Ranked by within-session accuracy:**

1. `eegnet` — within 0.6113
2. `deepconvnet` — within 0.6064
3. `fbcnet` — within 0.5535

## Cross-session drop table

| model | within Acc | cross Acc | drop (abs) | relative drop |
|---|---|---|---|---|
| `eegnet` | 0.6113 | 0.5381 | 0.0732 | 12.0% |
| `deepconvnet` | 0.6064 | 0.5363 | 0.0701 | 11.6% |
| `fbcnet` | 0.5535 | 0.5078 | 0.0457 | 8.2% |

> drop = within mean Acc − cross mean Acc; relative drop = 1 − Acc_cross/Acc_within.
