# Builds the XGBoost feature set and trains two models per dataset: the
# uncapped-RUL baseline (Option A ablation -- same model/features as the
# real one, only the target changes) and the real model on capped RUL.
# LSTM comes later.

import xgboost as xgb

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


if __name__ == "__main__":
    from src.features import (
        add_rolling_features,
        add_rul_targets,
        add_savgol_features,
        find_constant_sensors,
    )
    from src.load import DATASETS, load_rul, load_test, load_train
    from src.regimes import apply_regimes, fit_regimes
    from src.scoring import N_SPLITS, evaluate_at_final_cycle, group_kfold_cv, late_side_reduction_pct

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

    print(f"\nheadline: mean late-side penalty reduction across all four datasets: "
          f"{sum(late_reductions) / len(late_reductions):.2f}%")
    print(f"  per-dataset: {[round(r, 2) for r in late_reductions]}")
