# Builds the XGBoost feature set and trains two models per dataset: the
# uncapped-RUL baseline (Option A ablation -- same model/features as the
# real one, only the target changes) and the real model on capped RUL.
# Also builds the LSTM: sliding-window sequences, a variable-length model
# (so one trained model handles both full 30-cycle training windows and
# the shorter real sequences FD002/FD004 test units have), and a single
# unit-based validation split for early stopping rather than GroupKFold
# (the standard approach for neural nets; GroupKFold CV is already
# covered by XGBoost).

import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from src.load import SENSOR_COLS

# TF's default CPU ops are multi-threaded and not perfectly order-independent
# in floating point, so tf.random.set_seed alone doesn't make LSTM training
# byte-reproducible run to run. This pins it down, at a modest training-speed
# cost, so the fixed seed=42 means the same thing for the LSTM as it does
# everywhere else in this project.
tf.config.experimental.enable_op_determinism()

# Measured: FD001 LSTM test RMSE still varies run to run even with the
# above (15.50 / 14.79), and pinning intra/inter-op thread count to 1
# didn't close the gap either (15.77 / 15.35) -- some remaining op in this
# TF/Keras version pair isn't going through a deterministic path regardless
# (see docs/dataset-reference.md). Not chasing further: not a required
# output, and the thread pin was reverted since it cost training speed
# for no measured benefit. LSTM RMSE should be read as +/- ~0.5-1.0 run
# to run, not exact.

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


def make_xgb_model(**hyperparams) -> xgb.XGBRegressor:
    """hyperparams overrides XGBoost defaults -- per-dataset tuned values
    from notebooks/eda_tuning_fd002_fd004.ipynb, looked up via
    XGB_HYPERPARAMS. Empty for datasets that weren't tuned (FD001/FD003),
    which keeps them on the same untouched defaults as before."""
    return xgb.XGBRegressor(random_state=SEED, **hyperparams)


def train_uncapped_baseline(train_df, feature_cols: list[str], **hyperparams) -> xgb.XGBRegressor:
    """Ablation baseline: identical model and features to train_xgboost,
    trained on uncapped `rul` instead of `rul_capped`. Isolates the effect
    of the RUL cap itself, not algorithm choice. Takes the same hyperparams
    as train_xgboost so tuning doesn't also introduce a hyperparameter
    difference into that isolation."""
    model = make_xgb_model(**hyperparams)
    model.fit(train_df[feature_cols], train_df["rul"])
    return model


def train_xgboost(train_df, feature_cols: list[str], **hyperparams) -> xgb.XGBRegressor:
    model = make_xgb_model(**hyperparams)
    model.fit(train_df[feature_cols], train_df["rul_capped"])
    return model


# Per-dataset XGBoost hyperparameter overrides found in
# notebooks/eda_tuning_fd002_fd004.ipynb (RandomizedSearchCV, GroupKFold
# CV, FD002/FD004 only -- FD001/FD003 keep XGBoost's plain defaults, empty
# dict here). Improvement confirmed real against the per-dataset CV
# noise floor measured in eda_cv_folds.ipynb (FD002 ~0.94 RMSE gain vs. a
# ~0.32 fold-count noise floor; FD004 ~0.78 gain vs. a ~0.12 floor).
XGB_HYPERPARAMS: dict[str, dict] = {
    "FD001": {},
    "FD002": {
        "n_estimators": 500, "max_depth": 4, "learning_rate": 0.03,
        "subsample": 0.6, "colsample_bytree": 0.8, "min_child_weight": 3,
    },
    "FD003": {},
    "FD004": {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 5,
    },
}


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


def make_lstm_model(
    num_features: int,
    units: int = LSTM_UNITS,
    dropout: float = LSTM_DROPOUT,
    num_layers: int = 1,
    bidirectional: bool = False,
) -> keras.Model:
    """Variable-length time dimension (None, not a fixed 30) so the same
    trained weights handle both full training windows and the shorter real
    sequences short FD002/FD004 test units have -- Option A, no padding.
    num_layers > 1 stacks LSTM layers (each but the last returns full
    sequences, feeding the next one), tried in the FD002/FD004 tuning pass
    as a way to let the model build a regime-level representation before a
    degradation-level one. bidirectional wraps each LSTM layer to also read
    the window back to front -- legitimate here (not future-peeking) since
    each window is already a fixed, fully-known slice of the past by the
    time the model sees it; tried after research turned up a published
    bidirectional-LSTM result specifically beating plain/CNN-LSTM on FD002."""
    layers = [keras.layers.Input(shape=(None, num_features))]
    for i in range(num_layers):
        lstm_layer = keras.layers.LSTM(units, return_sequences=i < num_layers - 1)
        layers.append(keras.layers.Bidirectional(lstm_layer) if bidirectional else lstm_layer)
    layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(1, activation="relu"))  # RUL can't be negative
    model = keras.Sequential(layers)
    model.compile(optimizer="adam", loss="mse", metrics=[keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


def train_lstm(
    train_df,
    feature_cols: list[str],
    scaler: StandardScaler,
    target_col: str = "rul_capped",
    window: int = LSTM_WINDOW,
    units: int = LSTM_UNITS,
    dropout: float = LSTM_DROPOUT,
    num_layers: int = 1,
    bidirectional: bool = False,
    val_fraction: float = LSTM_VAL_FRACTION,
    seed: int = SEED,
) -> keras.Model:
    """Single train/validation split by unit (never by row), used for early
    stopping -- the standard approach for neural nets, not k-fold CV."""
    X, y, groups = make_sliding_windows(train_df, feature_cols, target_col, scaler, window)

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups))

    tf.random.set_seed(seed)
    model = make_lstm_model(
        num_features=X.shape[-1], units=units, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional
    )
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


# Per-dataset LSTM hyperparameter overrides (window/units/dropout/num_layers/
# bidirectional). Unlike XGB_HYPERPARAMS, all four are empty on purpose:
# notebooks/eda_tuning_fd002_fd004.ipynb tried 4 reasoned FD002/FD004
# variants (more units, a stacked layer, a shorter window, bidirectional)
# and none beat the plain defaults above, across two reruns -- a genuine
# negative result, not an unfinished search.
LSTM_HYPERPARAMS: dict[str, dict] = {
    "FD001": {},
    "FD002": {},
    "FD003": {},
    "FD004": {},
}

