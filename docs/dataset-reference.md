# C-MAPSS dataset reference

Lookup material for the NASA turbofan project. Read this when building the loader, the regime clustering, or the scoring module. Not needed for general work.

## File format

Source archive is CMAPSSData.zip, downloaded manually into `data/raw`. Not pip installable.

Space delimited, no header row. Trailing spaces on every line produce phantom NaN columns on read (two for train/test files, one for RUL files), so drop them explicitly rather than letting them through.

26 columns. unit id, cycle, 3 operational settings, 21 sensors.

## The four datasets

Measured via `python -m src.load`.

| Set | Train rows | Train units | Test rows | Test units | RUL rows | Conditions | Fault modes |
|---|---|---|---|---|---|---|---|
| FD001 | 20,631 | 100 | 13,096 | 100 | 100 | 1 | 1 |
| FD002 | 53,759 | 260 | 33,991 | 259 | 259 | 6 | 1 |
| FD003 | 24,720 | 100 | 16,596 | 100 | 100 | 1 | 2 |
| FD004 | 61,249 | 249 | 41,214 | 248 | 248 | 6 | 2 |

RUL row count matches the test unit count for that set, one true RUL value per test unit.

709 train engines + 707 test engines = 1,416 engines total.

Train and test are separate engine populations with independently numbered units: unit 1 in `train_FD002` and unit 1 in `test_FD002` are different physical engines that happen to share an id. Never group, join, or split by unit across train and test — unit id is only meaningful for grouping within one file.

## Constant channels

On FD001, sensors 1, 5, 6, 10, 16, 18, 19 are constant and drop out, along with operational setting 3. That leaves 14 usable channels.

Do not reuse that list on the other three. Re-detect constant columns per dataset by variance.

## Operating regimes

FD002 and FD004 have 6 flight conditions mixed together, which swamps the degradation signal if left alone.

K-means with k=6 on the three operational settings columns recovers them cleanly, since the conditions are discrete rather than continuous. Then standardize each sensor within its cluster so readings are comparable across regimes.

FD001 and FD003 have a single condition, so this step is a no-op there.

## Target

Max cycle minus current cycle, computed within each unit. Then a piecewise linear cap at 125.

The cap reflects that an engine at cycle 5 and the same engine at cycle 80 both look healthy. Without it the model tries to learn a linear countdown from a flat signal.

## Test evaluation

Train files run to failure. Test files stop part way through.

Each test set has a matching RUL_FDxxx.txt with one true RUL per unit, in unit order, corresponding to that unit's last recorded cycle.

So scoring means taking each test engine's final cycle, predicting once, and comparing to the RUL file. One prediction per engine, not per row.

## NASA asymmetric score

Let d be predicted RUL minus true RUL.

- Early prediction, d < 0, penalty is exp(-d/13) - 1
- Late prediction, d >= 0, penalty is exp(d/10) - 1

Summed across engines. Lower is better.

Late predictions are punished harder because late means the engine fails before maintenance happens. Frame model error as maintenance lead time, not as an abstract metric.

## Benchmark test RMSE

What published work reaches. Landing outside these ranges means something is wrong, in either direction.

| Set | Tree models | Deep models |
|---|---|---|
| FD001 | 17 to 19 | 12 to 13 |
| FD002 | 20 to 24 | 17 to 19 |
| FD003 | 17 to 19 | 12 to 13 |
| FD004 | 22 to 26 | 19 to 21 |

FD001 under 11 is the leakage tripwire, not a good result.

## Feature parameters

- Rolling mean, rolling standard deviation, rolling window slope, per unit per sensor
- Savitzky-Golay filter, window 11 to 21, polyorder 2 or 3, per unit per sensor
- LSTM sliding windows of 30 cycles. Engines shorter than 30 cycles need a padding or exclusion decision, make it explicitly

Measured via `df.groupby('unit')['cycle'].max()` on each loaded file. Every train-set unit across all four datasets has at least 128 cycles, so the under-30 case never comes up during training. It does come up at test time, because test files are truncated: FD001 test min is 31 cycles, FD003 test min is 38, both clear of the window. FD002 test has 6 units under 30 cycles (minimum 21), and FD004 test has 11 units under 30 cycles (minimum 19). So the padding-or-exclusion decision is only live for FD002 and FD004 at inference time, and needs to be made before the LSTM runs on those two.
