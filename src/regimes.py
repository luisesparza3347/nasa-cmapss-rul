# Detects the 6 flight-condition regimes mixed into FD002/FD004 and
# standardizes each sensor within its regime, so a reading means the same
# thing regardless of which condition produced it. FD001/FD003 have a
# single condition, so this step is a no-op there (regime is always 0).

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.load import SENSOR_COLS

SEED = 42
N_REGIMES = 6
OP_COLS = ["op1", "op2", "op3"]

# FD001/FD003 op-setting std is ~0.002 (sensor noise); FD002/FD004 regimes
# are separated by tens of units. Anything under 1.0 is clearly the former.
SINGLE_CONDITION_STD_THRESHOLD = 1.0


@dataclass
class RegimeFit:
    op_scaler: StandardScaler
    kmeans: KMeans
    sensor_scalers: dict[int, StandardScaler]


def _is_single_condition(df: pd.DataFrame) -> bool:
    return (df[OP_COLS].std() < SINGLE_CONDITION_STD_THRESHOLD).all()


def fit_regimes(
    train_df: pd.DataFrame, sensor_cols: list[str] = SENSOR_COLS
) -> tuple[pd.DataFrame, RegimeFit | None]:
    """Learn regime clusters and per-regime sensor scalers from training data only."""
    if _is_single_condition(train_df):
        out = train_df.copy()
        out["regime"] = 0
        return out, None

    op_scaler = StandardScaler().fit(train_df[OP_COLS])
    op_scaled = op_scaler.transform(train_df[OP_COLS])
    kmeans = KMeans(n_clusters=N_REGIMES, random_state=SEED, n_init=10).fit(op_scaled)

    out = train_df.copy()
    out["regime"] = kmeans.labels_
    out[sensor_cols] = out[sensor_cols].astype(float)

    sensor_scalers: dict[int, StandardScaler] = {}
    for regime, group in out.groupby("regime"):
        scaler = StandardScaler().fit(group[sensor_cols])
        sensor_scalers[regime] = scaler
        out.loc[group.index, sensor_cols] = scaler.transform(group[sensor_cols])

    return out, RegimeFit(op_scaler=op_scaler, kmeans=kmeans, sensor_scalers=sensor_scalers)


def apply_regimes(
    df: pd.DataFrame, fitted: RegimeFit | None, sensor_cols: list[str] = SENSOR_COLS
) -> pd.DataFrame:
    """Assign rows to already-learned regimes and rescale sensors with the already-learned scalers."""
    if fitted is None:
        out = df.copy()
        out["regime"] = 0
        return out

    op_scaled = fitted.op_scaler.transform(df[OP_COLS])
    out = df.copy()
    out["regime"] = fitted.kmeans.predict(op_scaled)
    out[sensor_cols] = out[sensor_cols].astype(float)

    for regime, group in out.groupby("regime"):
        scaler = fitted.sensor_scalers[regime]
        out.loc[group.index, sensor_cols] = scaler.transform(group[sensor_cols])

    return out


if __name__ == "__main__":
    from src.load import DATASETS, load_test, load_train

    for dataset in DATASETS:
        train = load_train(dataset)
        test = load_test(dataset)

        train_out, fitted = fit_regimes(train)
        test_out = apply_regimes(test, fitted)

        mode = "no-op (single condition)" if fitted is None else "k-means (6 regimes)"
        print(f"{dataset}: {mode}")
        print(f"  train regimes: {sorted(train_out['regime'].unique())}")
        print(f"  test  regimes: {sorted(test_out['regime'].unique())}")
