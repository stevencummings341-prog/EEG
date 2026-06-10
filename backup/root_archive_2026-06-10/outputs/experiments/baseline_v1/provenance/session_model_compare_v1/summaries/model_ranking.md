# Model ranking — within vs cross-session (baselines, 5-seed)

**Ranked by cross-session accuracy:**

1. `eegnet` — cross 0.7105, within 0.8067, drop 0.0961 (11.9%)
2. `deepconvnet` — cross 0.6811, within 0.7663, drop 0.0852 (11.1%)
3. `fbcnet` — cross 0.6280, within 0.7203, drop 0.0923 (12.8%)

**Ranked by within-session accuracy:**

1. `eegnet` — within 0.8067
2. `deepconvnet` — within 0.7663
3. `fbcnet` — within 0.7203

## Cross-session drop table

| model | within Acc | cross Acc | drop (abs) | relative drop |
|---|---|---|---|---|
| `eegnet` | 0.8067 | 0.7105 | 0.0961 | 11.9% |
| `deepconvnet` | 0.7663 | 0.6811 | 0.0852 | 11.1% |
| `fbcnet` | 0.7203 | 0.6280 | 0.0923 | 12.8% |

> drop = within mean Acc − cross mean Acc; relative drop = 1 − Acc_cross/Acc_within.
