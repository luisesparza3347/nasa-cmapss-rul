# Builds the XGBoost feature set and trains two models per dataset: the
# uncapped-RUL baseline (Option A ablation -- same model/features as the
# real one, only the target changes) and the real model on capped RUL.
# Also builds the LSTM: sliding-window sequences, a variable-length model
# (so one trained model handles both full 30-cycle training windows and
# the shorter real sequences FD002/FD004 test units have), and a single
# unit-based validation split for early stopping rather than GroupKFold
# (the standard approach for neural nets, and CLAUDE.md's CV requirement
# is already satisfied by XGBoost).

import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from src.load import SENSOR_COLS

SEED = 42

# cycle, operational settings, and the regime label aren't sensors, so the
# sensor feature-set decision doesn't cover them -- included for the same
# reason: XGBoost handles extra/redundant columns fine, so nothing is
# pre-filtered here. feature_importances_ after training is the empirical
# check, not a priori guessing.
EXTRA_FEATURE_COLS = ["cycle", "op1", "op2", "op3", "regime"]


def build_feature_cols(varying_sensors: list[str]) -> list[str]:
    """All columns fed to XGBoost: cycle, operational settings, regime
    label, and every representation of each non-constant sensor (raw,
    rolling mean/std/slope, Savitzky-Golay)."""
    cols = list(EXTRA_FEATURE_COLS)
    for s in varying_sensors:
        cols += [s, f"{s}_roll_mean", f"{s}_roll_std", f"{s}_roll_slope", f"{s}_savgol"]
    return cols


def make_xgb_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(random_state=SEED)


def train_uncapped_baseline(train_df, feature_cols: list[str]) -> xgb.XGBRegressor:
    """Ablation baseline: identical model and features to train_xgboost,
    trained on uncapped `rul` instead of `rul_capped`. Isolates the effect
    of the RUL cap, not algorithm choice, per the locked decision 3 cap."""
    model = make_xgb_model()
    model.fit(train_df[feature_cols], train_df["rul"])
    return model


def train_xgboost(train_df, feature_cols: list[str]) -> xgb.XGBRegressor:
    model = make_xgb_model()
    model.fit(train_df[feature_cols], train_df["rul_capped"])
    return model


# LSTM

LSTM_WINDOW = 30
LSTM_UNITS = 64
LSTM_DROPOUT = 0.2
LSTM_MAX_EPOCHS = 50
LSTM_PATIENCE = 5
LSTM_BATCH_SIZE = 64
LSTM_VAL_FRACTION = 0.2


def fit_lstm_scaler(train_df, feature_cols: list[str]) -> StandardScaler:
    """XGBoost is scale-invariant so its feature pipeline is untouched by
    this -- LSTM gates are not, and regimes.py only standardizes sensors
    for FD002/FD004, leaving FD001/FD003 sensors (and cycle/op1-3 for all
    four) in raw units. Fit on train only, same discipline as every other
    scaler in this project."""
    return StandardScaler().fit(train_df[feature_cols])


def make_sliding_windows(
    df, feature_cols: list[str], target_col: str, scaler: StandardScaler, window: int = LSTM_WINDOW
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-length window, training only: every train unit has >= 128
    cycles, so a window fits at every cycle >= window. Returns (X, y,
    groups) -- groups is the unit each window belongs to, for a group-based
    validation split."""
    scaled = df.copy()
    scaled[feature_cols] = scaler.transform(df[feature_cols])

    X_list, y_list, group_list = [], [], []
    for unit, group in scaled.sort_values(["unit", "cycle"]).groupby("unit"):
        values = group[feature_cols].to_numpy(dtype=np.float32)
        targets = group[target_col].to_numpy(dtype=np.float32)
        for end in range(window - 1, len(values)):
            X_list.append(values[end - window + 1: end + 1])
            y_list.append(targets[end])
            group_list.append(unit)

    return np.stack(X_list), np.array(y_list, dtype=np.float32), np.array(group_list)


def make_lstm_model(num_features: int, units: int = LSTM_UNITS, dropout: float = LSTM_DROPOUT) -> keras.Model:
    """Variable-length time dimension (None, not a fixed 30) so the same
    trained weights handle both full training windows and the shorter real
    sequences short FD002/FD004 test units have -- Option A, no padding."""
    model = keras.Sequential([
        keras.layers.Input(shape=(None, num_features)),
        keras.layers.LSTM(units),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1, activation="relu"),  # RUL can't be negative
    ])
    model.compile(optimizer="adam", loss="mse", metrics=[keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


def train_lstm(
    train_df,
    feature_cols: list[str],
    scaler: StandardScaler,
    target_col: str = "rul_capped",
    window: int = LSTM_WINDOW,
    val_fraction: float = LSTM_VAL_FRACTION,
    seed: int = SEED,
) -> keras.Model:
    """Single train/validation split by unit (never by row), used for early
    stopping -- the standard approach for neural nets, not k-fold CV."""
    X, y, groups = make_sliding_windows(train_df, feature_cols, target_col, scaler, window)

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups))

    tf.random.set_seed(seed)
    model = make_lstm_model(num_features=X.shape[-1])
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=LSTM_PATIENCE, restore_best_weights=True)

    model.fit(
        X[train_idx], y[train_idx],
        validation_data=(X[val_idx], y[val_idx]),
        epochs=LSTM_MAX_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )
    return model


if __name__ == "__main__":
    from src.features import (
        add_rolling_features,
        add_rul_targets,
        add_savgol_features,
        find_constant_sensors,
    )
    from src.load import DATASETS, load_rul, load_test, load_train
    from src.regimes import apply_regimes, fit_regimes
    from src.scoring import (
        N_SPLITS,
        evaluate_at_final_cycle,
        evaluate_sequence_model_at_final_cycle,
        group_kfold_cv,
        late_side_reduction_pct,
    )

    late_reductions = []

    for dataset in DATASETS:
        train = load_train(dataset)
        test = load_test(dataset)
        rul_true = load_rul(dataset)

        train_norm, fitted = fit_regimes(train)
        test_norm = apply_regimes(test, fitted)

        constant_sensors = find_constant_sensors(train_norm)
        varying_sensors = [s for s in SENSOR_COLS if s not in constant_sensors]

        train_feat = add_rolling_features(train_norm, varying_sensors)
        train_feat = add_savgol_features(train_feat, varying_sensors)
        train_feat = add_rul_targets(train_feat)

        test_feat = add_rolling_features(test_norm, varying_sensors)
        test_feat = add_savgol_features(test_feat, varying_sensors)

        feature_cols = build_feature_cols(varying_sensors)

        cv_rmses = group_kfold_cv(make_xgb_model, train_feat, feature_cols, "rul_capped")

        baseline_model = train_uncapped_baseline(train_feat, feature_cols)
        xgb_model = train_xgboost(train_feat, feature_cols)

        baseline_result = evaluate_at_final_cycle(baseline_model, test_feat, feature_cols, rul_true)
        xgb_result = evaluate_at_final_cycle(xgb_model, test_feat, feature_cols, rul_true)

        late_reduction = late_side_reduction_pct(
            baseline_result["y_true"], baseline_result["y_pred"], xgb_result["y_pred"]
        )
        late_reductions.append(late_reduction)

        print(f"{dataset}:")
        print(f"  {len(feature_cols)} feature columns ({len(varying_sensors)} varying sensors)")
        print(f"  GroupKFold(n_splits={N_SPLITS}) CV RMSE (capped target): "
              f"{[round(r, 2) for r in cv_rmses]}, mean={sum(cv_rmses) / len(cv_rmses):.2f}")
        print(f"  test @ final cycle -- uncapped baseline: "
              f"rmse={baseline_result['rmse']:.2f}, nasa_score={baseline_result['nasa_score']:.2f}")
        print(f"  test @ final cycle -- xgboost (capped):  "
              f"rmse={xgb_result['rmse']:.2f}, nasa_score={xgb_result['nasa_score']:.2f}")
        print(f"  late-side penalty reduction vs uncapped baseline: {late_reduction:.2f}%")

        lstm_scaler = fit_lstm_scaler(train_feat, feature_cols)
        lstm_model = train_lstm(train_feat, feature_cols, lstm_scaler)

        test_feat_scaled = test_feat.copy()
        test_feat_scaled[feature_cols] = lstm_scaler.transform(test_feat[feature_cols])
        lstm_result = evaluate_sequence_model_at_final_cycle(
            lstm_model, test_feat_scaled, feature_cols, rul_true, LSTM_WINDOW
        )
        print(f"  test @ final cycle -- lstm:               "
              f"rmse={lstm_result['rmse']:.2f}, nasa_score={lstm_result['nasa_score']:.2f}")

    print(f"\nheadline: mean late-side penalty reduction across all four datasets: "
          f"{sum(late_reductions) / len(late_reductions):.2f}%")
    print(f"  per-dataset: {[round(r, 2) for r in late_reductions]}")
