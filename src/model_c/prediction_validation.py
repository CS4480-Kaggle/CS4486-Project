"""Shared, fail-closed validation for Iceberg prediction and fold artifacts.

Reconstructed from the call sites in the supplied 3-code archive (not the
missing original module). Does not train models or modify predictions.

Expected file schemas:
    OOF:  id,fold,target,prediction,model_name,run_id
    Test: id,prediction,model_name,run_id
    Folds: id,target,fold
All validated outputs are returned in reference-ID order.
"""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "read_csv", "read_raw_json", "checked_ids", "checked_integers",
    "checked_probabilities", "validate_oof", "validate_test",
    "load_shared_folds", "csv_text", "sha256_file", "check_write", "atomic_write",
]


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV without silently altering ID strings or duplicating column names."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
    if not header.strip():
        raise ValueError(f"Empty CSV: {path}")
    # pandas automatically mangles duplicate headings; detect them before parsing.
    import csv
    columns = next(csv.reader([header.lstrip("\ufeff")]))
    if len(columns) != len(set(columns)):
        raise ValueError(f"Duplicate CSV columns in {path}: {columns}")
    return pd.read_csv(path, dtype={"id": "string", "run_id": "string", "model_name": "string"},
                       encoding="utf-8-sig", low_memory=False)


def read_raw_json(path: str | Path) -> pd.DataFrame:
    """Read train.json/test.json (JSON records) with their original row order."""
    path = Path(path)
    df = pd.read_json(path)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"Expected nonempty JSON record table: {path}")
    if "id" not in df:
        raise ValueError(f"JSON has no id field: {path}")
    return df


def _series(values, name: str) -> pd.Series:
    try:
        result = pd.Series(values).reset_index(drop=True)
    except Exception as exc:
        raise ValueError(f"{name} must be one-dimensional") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def checked_ids(values, name: str) -> np.ndarray:
    """Require nonmissing, distinct, nonempty IDs; never stringify nulls."""
    s = _series(values, name)
    if s.empty or s.isna().any():
        raise ValueError(f"{name}: IDs are missing or the table is empty")
    ids = s.astype(str)
    if ids.str.strip().eq("").any() or ids.str.lower().isin(["nan", "none", "null", "<na>"]).any():
        raise ValueError(f"{name}: blank or null-like ID")
    if ids.duplicated().any():
        raise ValueError(f"{name}: duplicate ID(s), e.g. {ids[ids.duplicated()].iloc[0]!r}")
    return ids.to_numpy(dtype=str)


def checked_integers(values, allowed: Iterable[int] | None, name: str) -> np.ndarray:
    """Validate integer-valued, finite entries and optional allowed set."""
    s = _series(values, name)
    if s.empty or s.isna().any():
        raise ValueError(f"{name}: missing integer(s) or empty input")
    try:
        numeric = pd.to_numeric(s, errors="raise").to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name}: nonnumeric integer value") from exc
    if not np.isfinite(numeric).all() or (numeric != np.floor(numeric)).any():
        raise ValueError(f"{name}: values must be finite integers")
    if (np.abs(numeric) > np.iinfo(np.int64).max).any():
        raise ValueError(f"{name}: integer overflow")
    result = numeric.astype(np.int64)
    if allowed is not None and not np.isin(result, list(allowed)).all():
        raise ValueError(f"{name}: unexpected values; allowed={list(allowed)}")
    return result


def checked_probabilities(values, name: str) -> np.ndarray:
    """Reject invalid probabilities before any clipping or metric calculation."""
    s = _series(values, name)
    if s.empty or s.isna().any():
        raise ValueError(f"{name}: missing probability or empty input")
    try:
        probs = pd.to_numeric(s, errors="raise").to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name}: nonnumeric probability") from exc
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError(f"{name}: probabilities must be finite numbers within [0, 1]")
    return probs


def _aligned(df: pd.DataFrame, reference_ids, required: list[str], name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name}: expected DataFrame")
    if df.columns.duplicated().any():
        raise ValueError(f"{name}: duplicate column names")
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns: {sorted(missing)}")
    expected = checked_ids(reference_ids, f"{name} reference")
    actual = checked_ids(df["id"], f"{name} file")
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError(f"{name}: IDs do not exactly match the reference set")
    # Reindex by reference IDs; prevents otherwise-valid shuffled CSV files from
    # silently assigning probabilities/labels to the wrong examples.
    out = df.copy()
    out["id"] = actual
    return out.set_index("id", drop=False).loc[expected].reset_index(drop=True)


def _validate_metadata(df: pd.DataFrame, run_id: str | None, name: str) -> None:
    for col in ("model_name", "run_id"):
        if col not in df:
            raise ValueError(f"{name}: missing {col} column")
        s = df[col]
        if s.isna().any() or s.empty:
            raise ValueError(f"{name}: missing {col}")
        v = s.astype(str)
        if (v.str.strip() == "").any() or v.nunique() != 1:
            raise ValueError(f"{name}: {col} must be a single nonblank value")
    if run_id is not None and str(df["run_id"].iloc[0]) != str(run_id):
        raise ValueError(f"{name}: run_id mismatch: expected {run_id!r}, got {df['run_id'].iloc[0]!r}")


def validate_oof(df: pd.DataFrame, train_ids, y, folds, run_id: str | None) -> pd.DataFrame:
    """Validate OOF metadata and values; return rows in train.json ID order."""
    result = _aligned(df, train_ids,
                      ["id", "fold", "target", "prediction", "model_name", "run_id"], "OOF")
    n = len(result)
    expected_y = checked_integers(y, [0, 1], "reference targets")
    expected_folds = checked_integers(folds, None, "reference folds")
    if len(expected_y) != n or len(expected_folds) != n:
        raise ValueError("OOF: reference targets/folds have wrong length")
    if (expected_folds < 0).any():
        raise ValueError("OOF: reference folds cannot be negative")
    if not np.array_equal(checked_integers(result["target"], [0, 1], "OOF targets"), expected_y):
        raise ValueError("OOF: target mismatch with train.json")
    if not np.array_equal(checked_integers(result["fold"], None, "OOF folds"), expected_folds):
        raise ValueError("OOF: fold mismatch with shared folds")
    checked_probabilities(result["prediction"], "OOF prediction")
    _validate_metadata(result, run_id, "OOF")
    return result


def validate_test(df: pd.DataFrame, test_ids, run_id: str | None) -> pd.DataFrame:
    """Validate test metadata and values; return rows in test.json ID order."""
    result = _aligned(df, test_ids, ["id", "prediction", "model_name", "run_id"], "test")
    checked_probabilities(result["prediction"], "test prediction")
    _validate_metadata(result, run_id, "test")
    return result


def load_shared_folds(train: pd.DataFrame, path: str | Path,
                      legacy_path: str | Path | None = None) -> pd.DataFrame:
    """Validate shared folds against train.json; check legacy copy if present.

    Returns id,target,fold (plus any original extra columns) in train.json order.
    Never creates or silently replaces folds.
    """
    if "id" not in train or "is_iceberg" not in train:
        raise ValueError("Training data must contain id and is_iceberg")
    train_ids = checked_ids(train["id"], "train.json")
    y = checked_integers(train["is_iceberg"], [0, 1], "train.json targets")

    def load_one(file: str | Path) -> pd.DataFrame:
        table = _aligned(read_csv(file), train_ids, ["id", "target", "fold"], str(file))
        if not np.array_equal(checked_integers(table["target"], [0, 1], f"{file} targets"), y):
            raise ValueError(f"{file}: targets disagree with train.json")
        fold = checked_integers(table["fold"], None, f"{file} folds")
        if (fold < 0).any() or len(np.unique(fold)) < 2:
            raise ValueError(f"{file}: folds must be nonnegative and include at least two splits")
        if not np.array_equal(np.unique(fold), np.arange(fold.max() + 1)):
            raise ValueError(f"{file}: fold numbers must be contiguous starting at zero")
        table["fold"] = fold
        table["target"] = y
        return table

    shared = load_one(path)
    if legacy_path is not None and Path(legacy_path).exists():
        old = load_one(legacy_path)
        if not np.array_equal(shared["fold"].to_numpy(), old["fold"].to_numpy()):
            raise ValueError(f"Shared folds {path} and legacy folds {legacy_path} disagree")
    return shared


def csv_text(df: pd.DataFrame) -> str:
    """Serialize without row index; preserve numeric prediction values."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("csv_text expects a DataFrame")
    return df.to_csv(index=False, lineterminator="\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_write(path: str | Path, contents: str, overwrite: bool = False) -> str:
    """Verify a planned write; refuse replacing different bytes without opt-in."""
    path = Path(path)
    if not isinstance(contents, str):
        raise TypeError("contents must be text")
    if not path.exists():
        return "created"
    if not path.is_file():
        raise ValueError(f"Cannot write over a non-file path: {path}")
    if path.read_bytes() == contents.encode("utf-8"):
        return "already_identical"
    if not overwrite:
        raise FileExistsError(f"Refusing to replace different file without --overwrite: {path}")
    return "overwritten"


def atomic_write(path: str | Path, contents: str) -> None:
    """Atomically replace one file; caller must first run check_write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
            temp_path = f.name
            f.write(contents)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)
