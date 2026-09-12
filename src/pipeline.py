# The only script in src/: ties load -> regimes -> features -> models ->
# scoring together for one dataset or all four, writes metrics and
# diagnostic figures to disk, and prints the two required outputs
# (FD001 final-model test RMSE, headline reduction in late predictions %)
# explicitly.

import argparse
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import (
    add_rolling_features,
    add_rul_targets,
    add_savgol_features,
    find_constant_sensors,
)
from src.load import DATASETS, SENSOR_COLS, load_rul, load_test, load_train
from src.models import (
    LSTM_HYPERPARAMS,
    LSTM_WINDOW,
    XGB_HYPERPARAMS,
    build_feature_cols,
    fit_lstm_scaler,
    make_xgb_model,
    train_lstm,
    train_uncapped_baseline,
    train_xgboost,
)
from src.regimes import apply_regimes, fit_regimes
from src.scoring import (
    evaluate_at_final_cycle,
    evaluate_sequence_model_at_final_cycle,
    group_kfold_cv,
    late_prediction_count_reduction_pct,
    late_prediction_pct,
    late_side_reduction_pct,
)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
METRICS_PATH = RESULTS_DIR / "metrics.csv"
LATE_REDUCTION_PATH = RESULTS_DIR / "late_side_reduction.csv"

FINAL_MODEL = "lstm"


def run_dataset(dataset: str) -> dict:
    """Run one dataset end to end. Returns metric rows plus the raw
    predictions figures are built from."""
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

    xgb_hyperparams = XGB_HYPERPARAMS.get(dataset, {})
    lstm_hyperparams = LSTM_HYPERPARAMS.get(dataset, {})
    lstm_window = lstm_hyperparams.get("window", LSTM_WINDOW)

    cv_rmses = group_kfold_cv(
        partial(make_xgb_model, **xgb_hyperparams), train_feat, feature_cols, "rul_capped"
    )

    baseline_model = train_uncapped_baseline(train_feat, feature_cols, **xgb_hyperparams)
    xgb_model = train_xgboost(train_feat, feature_cols, **xgb_hyperparams)

    baseline_result = evaluate_at_final_cycle(baseline_model, test_feat, feature_cols, rul_true)
    xgb_result = evaluate_at_final_cycle(xgb_model, test_feat, feature_cols, rul_true)

    penalty_reduction_pct = late_side_reduction_pct(
        baseline_result["y_true"], baseline_result["y_pred"], xgb_result["y_pred"]
    )
    count_reduction_pct = late_prediction_count_reduction_pct(
        baseline_result["y_true"], baseline_result["y_pred"], xgb_result["y_pred"]
    )
    late_pct_baseline = late_prediction_pct(baseline_result["y_true"], baseline_result["y_pred"])
    late_pct_capped = late_prediction_pct(xgb_result["y_true"], xgb_result["y_pred"])

    lstm_scaler = fit_lstm_scaler(train_feat, feature_cols)
    lstm_model = train_lstm(train_feat, feature_cols, lstm_scaler, **lstm_hyperparams)

    test_feat_scaled = test_feat.copy()
    test_feat_scaled[feature_cols] = lstm_scaler.transform(test_feat[feature_cols])
    lstm_result = evaluate_sequence_model_at_final_cycle(
        lstm_model, test_feat_scaled, feature_cols, rul_true, lstm_window
    )

    # rmse/nasa_score below are the raw-label numbers (scored against
    # RUL_FDxxx.txt as-is) -- the additional *_label / *_subset columns are
    # the other two labeling-convention views from _labeling_convention_metrics,
    # not a "which one is right" pick (see docs/Log.md, 2026-09-12 investigation).
    def _test_row(model_name: str, result: dict) -> dict:
        return {
            "dataset": dataset, "model": model_name, "split": "test",
            "rmse": result["rmse"], "nasa_score": result["nasa_score"],
            "rmse_capped_label": result["rmse_capped_label"],
            "nasa_score_capped_label": result["nasa_score_capped_label"],
            "rmse_below_cap_subset": result["rmse_below_cap_subset"],
            "nasa_score_below_cap_subset": result["nasa_score_below_cap_subset"],
            "n_below_cap_subset": result["n_below_cap_subset"],
            "pct_below_cap_subset": result["pct_below_cap_subset"],
        }

    metric_rows = [
        {"dataset": dataset, "model": "xgboost", "split": "cv",
         "rmse": sum(cv_rmses) / len(cv_rmses), "nasa_score": np.nan},
        _test_row("xgboost_baseline_uncapped", baseline_result),
        _test_row("xgboost", xgb_result),
        _test_row("lstm", lstm_result),
    ]

    return {
        "metric_rows": metric_rows,
        "late_penalty_reduction_pct": penalty_reduction_pct,
        "late_count_reduction_pct": count_reduction_pct,
        "late_pct_baseline_uncapped": late_pct_baseline,
        "late_pct_capped": late_pct_capped,
        "lstm_y_true": lstm_result["y_true"],
        "lstm_y_pred": lstm_result["y_pred"],
    }


def _upsert_rows(path: Path, new_rows: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Replace any existing rows matching new_rows' keys, keep everything
    else, write back to disk. Lets a single-dataset run and --all compose
    without duplicating or clobbering other datasets' rows."""
    if path.exists():
        existing = pd.read_csv(path)
        new_keys = set(new_rows[key_cols].apply(tuple, axis=1))
        existing = existing[~existing[key_cols].apply(tuple, axis=1).isin(new_keys)]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined = combined.sort_values(key_cols).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def update_late_reduction(dataset: str, late_stats: dict) -> pd.DataFrame:
    row = pd.DataFrame([{
        "dataset": dataset,
        "late_pct_baseline_uncapped": late_stats["late_pct_baseline_uncapped"],
        "late_pct_capped": late_stats["late_pct_capped"],
        "late_count_reduction_pct": late_stats["late_count_reduction_pct"],
        "late_penalty_reduction_pct": late_stats["late_penalty_reduction_pct"],
    }])
    combined = _upsert_rows(LATE_REDUCTION_PATH, row, key_cols=["dataset"])

    real = combined[combined["dataset"] != "headline_mean"]
    if set(DATASETS).issubset(set(real["dataset"])):
        headline_row = pd.DataFrame([{
            "dataset": "headline_mean",
            "late_pct_baseline_uncapped": real["late_pct_baseline_uncapped"].mean(),
            "late_pct_capped": real["late_pct_capped"].mean(),
            "late_count_reduction_pct": real["late_count_reduction_pct"].mean(),
            "late_penalty_reduction_pct": real["late_penalty_reduction_pct"].mean(),
        }])
        combined = _upsert_rows(LATE_REDUCTION_PATH, headline_row, key_cols=["dataset"])

    return combined


def _plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, dataset: str, model: str, path: Path) -> None:
    order = np.argsort(y_true)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(y_true[order], label="true RUL", marker="o", linestyle="", alpha=0.6, ms=4)
    ax.plot(y_pred[order], label="predicted RUL", marker="x", linestyle="", alpha=0.6, ms=4)
    ax.set_xlabel(f"{dataset} test engine, sorted by true RUL")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"{dataset}: {model} predicted vs. true RUL at final test cycle")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_late_reduction(late_df: pd.DataFrame, path: Path) -> None:
    real = late_df[late_df["dataset"] != "headline_mean"].sort_values("dataset")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(real["dataset"], real["late_count_reduction_pct"])
    ax.set_ylabel("reduction in late predictions (%)")
    ax.set_title("Capped vs. uncapped RUL: reduction in share of late predictions")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _print_metric_rows(metric_rows: list[dict]) -> None:
    for row in metric_rows:
        score = f", nasa_score={row['nasa_score']:.2f}" if pd.notna(row["nasa_score"]) else ""
        print(f"  {row['model']:<28} {row['split']:<4} rmse={row['rmse']:.2f}{score}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the C-MAPSS RUL pipeline end to end.")
    parser.add_argument("dataset", nargs="?", choices=DATASETS, help="single dataset to run, e.g. FD001")
    parser.add_argument("--all", action="store_true", help="run all four datasets")
    args = parser.parse_args()

    if args.all:
        datasets = list(DATASETS)
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.error("specify a dataset (FD001/FD002/FD003/FD004) or --all")

    for dataset in datasets:
        print(f"=== {dataset} ===")
        result = run_dataset(dataset)

        _upsert_rows(METRICS_PATH, pd.DataFrame(result["metric_rows"]), key_cols=["dataset", "model", "split"])
        late_df = update_late_reduction(dataset, result)

        _print_metric_rows(result["metric_rows"])
        print(f"  late predictions: {result['late_pct_baseline_uncapped']:.1f}% (uncapped baseline) -> "
              f"{result['late_pct_capped']:.1f}% (capped), a {result['late_count_reduction_pct']:.2f}% reduction")

        if dataset == "FD001":
            _plot_pred_vs_true(
                result["lstm_y_true"], result["lstm_y_pred"], "FD001", FINAL_MODEL,
                FIGURES_DIR / "fd001_pred_vs_true.png",
            )

        if set(DATASETS).issubset(set(late_df[late_df["dataset"] != "headline_mean"]["dataset"])):
            _plot_late_reduction(late_df, FIGURES_DIR / "late_side_reduction.png")

    print("\n=== required outputs ===")

    metrics_df = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()
    fd001_final = metrics_df[
        (metrics_df["dataset"] == "FD001") & (metrics_df["model"] == FINAL_MODEL) & (metrics_df["split"] == "test")
    ]
    if not fd001_final.empty:
        print(f"1. FD001 test RMSE, final model ({FINAL_MODEL}): {fd001_final['rmse'].iloc[0]:.2f}")
    else:
        print("1. FD001 not yet run -- required output 1 unavailable")

    late_df = pd.read_csv(LATE_REDUCTION_PATH) if LATE_REDUCTION_PATH.exists() else pd.DataFrame()
    headline = late_df[late_df.get("dataset") == "headline_mean"] if not late_df.empty else late_df
    if not headline.empty:
        print(f"2. Reduction in late predictions, headline mean across datasets: "
              f"{headline['late_count_reduction_pct'].iloc[0]:.2f}%")
    else:
        print("2. Not all four datasets run yet -- headline reduction unavailable")


if __name__ == "__main__":
    main()
