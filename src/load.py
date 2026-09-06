# Reads the raw C-MAPSS train/test/RUL files, assigns column names, and
# strips the phantom NaN columns that trailing spaces produce on read.

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
DATASETS = ("FD001", "FD002", "FD003", "FD004")

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + SENSOR_COLS


def _read_whitespace_file(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a space-delimited, headerless CMAPSS file and drop the phantom
    trailing-space columns pandas produces from `sep=' '`."""
    raw = pd.read_csv(path, sep=" ", header=None)

    assert raw.shape[1] >= len(columns), (
        f"{path.name}: expected at least {len(columns)} raw columns, got {raw.shape[1]}"
    )

    phantom_cols = raw.columns[len(columns):]
    assert raw[phantom_cols].isna().all().all(), (
        f"{path.name}: trailing columns beyond position {len(columns)} "
        f"are not all-NaN, refusing to drop them"
    )

    df = raw.iloc[:, : len(columns)].copy()
    df.columns = columns

    assert df.shape[1] == len(columns), f"{path.name}: column count mismatch after assigning names"
    assert not df.isna().any().any(), f"{path.name}: unexpected NaNs remain in kept columns"

    return df


def load_train(dataset: str) -> pd.DataFrame:
    path = RAW_DIR / f"train_{dataset}.txt"
    return _read_whitespace_file(path, COLUMNS)


def load_test(dataset: str) -> pd.DataFrame:
    path = RAW_DIR / f"test_{dataset}.txt"
    return _read_whitespace_file(path, COLUMNS)


def load_rul(dataset: str) -> pd.Series:
    """One true RUL value per unit, in unit order, for that unit's last
    recorded cycle in the matching test file. Not derivable from the test
    data itself, since test files are truncated mid-life."""
    path = RAW_DIR / f"RUL_{dataset}.txt"
    df = _read_whitespace_file(path, ["RUL"])
    return df["RUL"]


def load_all_train() -> dict[str, pd.DataFrame]:
    return {dataset: load_train(dataset) for dataset in DATASETS}


def load_all_test() -> dict[str, pd.DataFrame]:
    return {dataset: load_test(dataset) for dataset in DATASETS}


def load_all_rul() -> dict[str, pd.Series]:
    return {dataset: load_rul(dataset) for dataset in DATASETS}


if __name__ == "__main__":
    for dataset in DATASETS:
        train = load_train(dataset)
        test = load_test(dataset)
        rul = load_rul(dataset)

        assert rul.shape[0] == test["unit"].nunique(), (
            f"{dataset}: RUL row count ({rul.shape[0]}) does not match "
            f"test unit count ({test['unit'].nunique()})"
        )

        print(f"{dataset}:")
        print(f"  train  rows={train.shape[0]:>7}  units={train['unit'].nunique():>4}")
        print(f"  test   rows={test.shape[0]:>7}  units={test['unit'].nunique():>4}")
        print(f"  rul    rows={rul.shape[0]:>7}")
