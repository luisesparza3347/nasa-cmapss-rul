# Builds the RUL target (capped and uncapped) and the per-sensor engineered
# features (rolling mean/std/slope, Savitzky-Golay smoothing) that feed the
# models. Drops sensors that are constant within every engine's life.

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.load import SENSOR_COLS

RUL_CAP = 125
ROLLING_WINDOW = 15
SAVGOL_WINDOW = 21
SAVGOL_POLYORDER = 2

# Measured (notebooks/eda_regimes.ipynb): the constant-sensor set differs per
# dataset. FD001's tightest real-signal gap is at 0.037, its highest constant
# sensor at 0.0029 -- 0.01 sits well inside every dataset's gap.
CONSTANT_STD_THRESHOLD = 0.01


def add_rul_targets(train_df: pd.DataFrame) -> pd.DataFrame:
    """Uncapped and piecewise-linear-capped RUL, train data only (test RUL
    comes from RUL_FDxxx.txt in scoring.py, not from this data)."""
    out = train_df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = max_cycle - out["cycle"]
    out["rul_capped"] = out["rul"].clip(upper=RUL_CAP)
    return out


def find_constant_sensors(
    train_df: pd.DataFrame, sensor_cols: list[str] = SENSOR_COLS, threshold: float = CONSTANT_STD_THRESHOLD
) -> list[str]:
    """Sensors whose worst-case (max) per-unit std, on train data, is at or
    below threshold. Fit on train only; reuse this exact list for test."""
    max_per_unit_std = train_df.groupby("unit")[sensor_cols].std().max()
    return list(max_per_unit_std[max_per_unit_std <= threshold].index)


def _rolling_slope(values: np.ndarray) -> float:
    """Least-squares slope of values against 0..n-1. Same math as
    np.polyfit(x, values, 1)[0], but ~2.6x faster measured on FD004 (61K
    rows x 17 sensors): np.polyfit solves a general polynomial fit via SVD,
    which is unnecessary work for a plain degree-1 slope."""
    n = len(values)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=np.float64)
    dx = x - x.mean()
    denom = (dx * dx).sum()
    if denom == 0:
        return 0.0
    return (dx * (values - values.mean())).sum() / denom


def add_rolling_features(df: pd.DataFrame, sensor_cols: list[str], window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Rolling mean/std/slope per unit per sensor. Mean is defined from a
    single point (min_periods=1); std and slope need at least 2, so their
    first-cycle NaN is filled with 0 -- both XGBoost and the LSTM consume
    this output, and unlike XGBoost, the LSTM can't tolerate NaN inputs."""
    out = df.sort_values(["unit", "cycle"]).copy()
    grouped = out.groupby("unit")

    for s in sensor_cols:
        out[f"{s}_roll_mean"] = grouped[s].transform(lambda g: g.rolling(window, min_periods=1).mean())
        out[f"{s}_roll_std"] = grouped[s].transform(lambda g: g.rolling(window, min_periods=2).std()).fillna(0)
        out[f"{s}_roll_slope"] = (
            grouped[s]
            .transform(lambda g: g.rolling(window, min_periods=2).apply(_rolling_slope, raw=True, engine="numba"))
            .fillna(0)
        )

    return out.sort_index()


def _savgol_for_unit(y: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    n = len(y)
    w = window if n >= window else (n if n % 2 == 1 else n - 1)
    return savgol_filter(y, window_length=w, polyorder=polyorder)


def add_savgol_features(
    df: pd.DataFrame, sensor_cols: list[str], window: int = SAVGOL_WINDOW, polyorder: int = SAVGOL_POLYORDER
) -> pd.DataFrame:
    """Savitzky-Golay smoothed sensor value per unit per sensor. Shrinks the
    window to the largest odd number <= a unit's trajectory length, since
    some FD002/FD004 test units are shorter than the default window."""
    out = df.sort_values(["unit", "cycle"]).copy()
    grouped = out.groupby("unit")

    for s in sensor_cols:
        out[f"{s}_savgol"] = grouped[s].transform(lambda g: _savgol_for_unit(g.to_numpy(), window, polyorder))

    return out.sort_index()


if __name__ == "__main__":
    from src.load import DATASETS, load_test, load_train
    from src.regimes import apply_regimes, fit_regimes

    for dataset in DATASETS:
        train = load_train(dataset)
        test = load_test(dataset)

        train_norm, fitted = fit_regimes(train)
        test_norm = apply_regimes(test, fitted)

        constant_sensors = find_constant_sensors(train_norm)
        varying_sensors = [s for s in SENSOR_COLS if s not in constant_sensors]

        train_feat = add_rolling_features(train_norm, varying_sensors)
        train_feat = add_savgol_features(train_feat, varying_sensors)
        train_feat = add_rul_targets(train_feat)

        test_feat = add_rolling_features(test_norm, varying_sensors)
        test_feat = add_savgol_features(test_feat, varying_sensors)

        feature_cols = [c for c in train_feat.columns if c not in ("unit", "cycle", "regime", "rul", "rul_capped")]
        nan_count = train_feat[feature_cols].isna().sum().sum()

        print(f"{dataset}: dropped {len(constant_sensors)} constant sensors {constant_sensors}")
        print(f"  kept {len(varying_sensors)} sensors, {len(feature_cols)} feature columns")
        print(f"  train rul range: {train_feat['rul'].min()}-{train_feat['rul'].max()}, "
              f"rul_capped range: {train_feat['rul_capped'].min()}-{train_feat['rul_capped'].max()}")
        print(f"  NaNs remaining in train features: {nan_count}")
        print(f"  test feature rows: {len(test_feat)}, NaNs: {test_feat[feature_cols].isna().sum().sum()}")
