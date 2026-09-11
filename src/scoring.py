# RMSE, the NASA asymmetric scoring function, and the two evaluation paths
# CLAUDE.md requires kept separate: GroupKFold cross-validation on training
# data, and one-prediction-per-engine scoring at each test unit's final
# recorded cycle against RUL_FDxxx.txt.

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Middle ground between 5 (cheaper, standard) and 10 (more stable, costlier)
# folds, picked before models.py existed to run a real sensitivity check.
# Revisit once that check (train at a couple of k values, compare how much
# CV RMSE varies across folds) can be run.
N_SPLITS = 7

# GroupKFold has no random_state -- its split is a deterministic assignment
# of groups to folds (no shuffling), not a randomized one. The seed=42 rule
# still applies to the model trained inside each fold.
SEED = 42


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """Asymmetric penalty from dataset-reference.md: d = predicted - true.
    Early (d < 0) is penalized gently, late (d >= 0) is penalized harshly,
    since a late prediction means the engine fails before maintenance."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    d = y_pred - y_true
    penalty = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(penalty.sum())


def late_side_penalty_sum(y_true, y_pred) -> float:
    """Sum of the NASA score's late-penalty term (d = predicted - true >= 0)
    only, ignoring early predictions entirely. Captures total late-failure
    risk exposure -- both how often the model is late and how badly -- not
    average severity conditional on being late."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    d = y_pred - y_true
    late_d = d[d >= 0]
    return float((np.exp(late_d / 10) - 1).sum())


def late_side_reduction_pct(y_true, y_pred_baseline, y_pred_final) -> float:
    """Percentage drop in summed late-side penalty, final model vs the
    uncapped-RUL baseline (required output 2 in CLAUDE.md). Positive means
    the final model carries less total late-failure risk."""
    baseline_penalty = late_side_penalty_sum(y_true, y_pred_baseline)
    final_penalty = late_side_penalty_sum(y_true, y_pred_final)

    assert baseline_penalty > 0, "baseline has zero late-side penalty -- reduction % is undefined"

    return (baseline_penalty - final_penalty) / baseline_penalty * 100


def last_cycle_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unit: its last recorded cycle. Test files are truncated
    mid-life, so this is the row the true RUL in RUL_FDxxx.txt refers to."""
    return (
        df.sort_values(["unit", "cycle"])
        .groupby("unit", as_index=False)
        .tail(1)
        .sort_values("unit")
        .reset_index(drop=True)
    )


def evaluate_at_final_cycle(
    model, test_df: pd.DataFrame, feature_cols: list[str], rul_true: pd.Series
) -> dict:
    """One prediction per engine, at its final cycle, scored against the
    matching RUL_FDxxx.txt. Not row-by-row -- that would score cycles a
    maintenance decision was never actually made at."""
    last_rows = last_cycle_per_unit(test_df)

    assert len(last_rows) == len(rul_true), (
        f"last-cycle row count ({len(last_rows)}) does not match "
        f"RUL row count ({len(rul_true)})"
    )

    y_true = rul_true.reset_index(drop=True).to_numpy()
    y_pred = model.predict(last_rows[feature_cols])

    return {
        "rmse": rmse(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def last_window_per_unit(df: pd.DataFrame, feature_cols: list[str], window: int) -> list[np.ndarray]:
    """Per unit, its last up-to-`window` cycles of feature values, in
    order, unit ascending (matching RUL_FDxxx.txt order). Shorter than
    `window` only for engines truncated below it -- fed to the model as-is,
    no padding (Option A)."""
    sequences = []
    for unit, group in df.sort_values(["unit", "cycle"]).groupby("unit"):
        sequences.append((unit, group[feature_cols].to_numpy(dtype=np.float32)[-window:]))
    sequences.sort(key=lambda pair: pair[0])
    return [seq for _, seq in sequences]


def evaluate_sequence_model_at_final_cycle(
    model, test_df: pd.DataFrame, feature_cols: list[str], rul_true: pd.Series, window: int
) -> dict:
    """LSTM equivalent of evaluate_at_final_cycle: one prediction per
    engine from its real (possibly shorter-than-window) final sequence,
    scored against RUL_FDxxx.txt. test_df must already be scaled with the
    same scaler the model was trained with."""
    sequences = last_window_per_unit(test_df, feature_cols, window)

    assert len(sequences) == len(rul_true), (
        f"sequence count ({len(sequences)}) does not match RUL row count ({len(rul_true)})"
    )

    y_true = rul_true.reset_index(drop=True).to_numpy()
    y_pred = np.array([
        model.predict(seq[np.newaxis, :, :], verbose=0)[0, 0] for seq in sequences
    ])

    return {
        "rmse": rmse(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def group_kfold_cv(
    estimator_fn,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_splits: int = N_SPLITS,
) -> list[float]:
    """RMSE per fold, splitting by unit so a single engine's history never
    lands on both sides of a fold. For cross-validation inside the training
    data only -- test scoring is evaluate_at_final_cycle, not this."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["unit"]

    fold_rmses = []
    for train_idx, val_idx in gkf.split(df[feature_cols], df[target_col], groups):
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]

        model = estimator_fn()
        model.fit(train_fold[feature_cols], train_fold[target_col])
        y_pred = model.predict(val_fold[feature_cols])

        fold_rmses.append(rmse(val_fold[target_col], y_pred))

    return fold_rmses


if __name__ == "__main__":
    # Sanity check against dataset-reference.md's worked description: a
    # perfect prediction scores 0 on both metrics, and a late prediction
    # is penalized more than an equally-sized early one.
    y_true = np.array([50.0, 50.0, 50.0])
    y_pred_perfect = np.array([50.0, 50.0, 50.0])
    y_pred_early = np.array([50.0, 50.0, 40.0])
    y_pred_late = np.array([50.0, 50.0, 60.0])

    assert rmse(y_true, y_pred_perfect) == 0.0
    assert nasa_score(y_true, y_pred_perfect) == 0.0

    early_penalty = nasa_score(y_true, y_pred_early)
    late_penalty = nasa_score(y_true, y_pred_late)
    assert late_penalty > early_penalty, "a late prediction must be penalized harder than an equal-sized early one"

    print("rmse/nasa_score sanity checks passed")
    print(f"  perfect: rmse={rmse(y_true, y_pred_perfect)}, nasa_score={nasa_score(y_true, y_pred_perfect)}")
    print(f"  10-cycle early: nasa_score={early_penalty:.4f}")
    print(f"  10-cycle late:  nasa_score={late_penalty:.4f}")

    # An all-early baseline has zero late-side penalty; a baseline with one
    # late miss should show 100% reduction once the final model fixes it.
    assert late_side_penalty_sum(y_true, y_pred_early) == 0.0
    baseline_one_late = np.array([50.0, 50.0, 90.0])  # one engine 40 cycles late
    final_fixed = np.array([50.0, 50.0, 50.0])  # same engine, now on time
    reduction = late_side_reduction_pct(y_true, baseline_one_late, final_fixed)
    assert reduction == 100.0
    print(f"  late-side reduction sanity check (baseline has 1 late miss, final fixes it): {reduction:.1f}%")
