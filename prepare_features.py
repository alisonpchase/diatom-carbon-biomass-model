# prepare_features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


def _require_pandas(df):
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise ImportError("prepare_features requires pandas DataFrames.") from e
    if not hasattr(df, "loc") or not hasattr(df, "columns"):
        raise TypeError("Expected a pandas DataFrame.")
    return df


@dataclass(frozen=True)
class PrepConfig:
    # These are the FINAL feature columns expected by the model (after derivations)
    feature_names: List[str]

    # Variables to log10 in the feature space (after ratios are computed)
    var_to_log: List[str]

    # Optional: training target config
    target_name: Optional[str] = None
    target_log10: bool = False

    # Ratio definitions: output_feature -> (numerator_col, denominator_col)
    ratio_defs: Optional[Dict[str, tuple[str, str]]] = None

    # If you want to treat zero/negative as invalid (recommended for log10 vars)
    require_positive_for_log: bool = True


@dataclass(frozen=True)
class PrepResult:
    df_features: Any                 # pandas.DataFrame
    valid_mask: np.ndarray           # for original df rows (unless drop_invalid=True in training wrapper)
    reasons: Dict[str, np.ndarray]   # reason -> boolean mask


def _log10_series_safe(s):
    s2 = s.where(s > 0, np.nan)
    return np.log10(s2)


def compute_derived_features(df, cfg: PrepConfig):
    """
    Compute derived features (e.g., pigment ratios) in-place on a copy of df.
    Returns df with new columns added.
    """
    df = _require_pandas(df).copy()

    if cfg.ratio_defs:
        for out_col, (num_col, den_col) in cfg.ratio_defs.items():
            if out_col in df.columns:
                # Don't overwrite unless you want to; safest is to overwrite to guarantee consistency
                pass

            if num_col not in df.columns:
                raise ValueError(f"Missing numerator column '{num_col}' needed to compute '{out_col}'")
            if den_col not in df.columns:
                raise ValueError(f"Missing denominator column '{den_col}' needed to compute '{out_col}'")

            num = df[num_col]
            den = df[den_col]

            # Compute ratio; division by 0 -> inf -> later becomes NaN and invalid
            df[out_col] = num / den

    return df


def prepare_features_only(df, cfg: PrepConfig) -> PrepResult:
    """
    Prepare X features:
      - compute derived ratios (if needed)
      - select & order cfg.feature_names
      - replace ±inf with NaN
      - log10 transform cfg.var_to_log (safe, nonpositive -> NaN)
      - build valid_mask
    """
    df = compute_derived_features(df, cfg)

    feature_names = list(cfg.feature_names)
    var_to_log = list(cfg.var_to_log)

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns AFTER derivation: {missing}")

    Xdf = df.loc[:, feature_names].copy()
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)

    # Flag non-positive values for log vars (in raw space, before log)
    if var_to_log and cfg.require_positive_for_log:
        nonpositive_log = (Xdf[var_to_log] <= 0).any(axis=1).to_numpy()
    else:
        nonpositive_log = np.zeros(len(Xdf), dtype=bool)

    if var_to_log:
        for c in var_to_log:
            Xdf[c] = _log10_series_safe(Xdf[c])

    any_nan = Xdf.isna().any(axis=1).to_numpy()
    any_nonfinite = (~np.isfinite(Xdf.to_numpy(dtype=float, copy=False))).any(axis=1)

    valid = ~(any_nan | any_nonfinite)

    reasons = {
        "nonpositive_in_log_vars": nonpositive_log,
        "nan_in_features": any_nan,
        "nonfinite_in_features": any_nonfinite,
    }

    return PrepResult(df_features=Xdf, valid_mask=valid, reasons=reasons)


def prepare_training_xy(df, cfg: PrepConfig, *, drop_invalid: bool = True):
    """
    Prepare X and y for training, including derived ratios + log transforms.
    Returns Xdf, y_series, PrepResult.
    """
    if cfg.target_name is None:
        raise ValueError("cfg.target_name must be set for training prep.")

    df = compute_derived_features(df, cfg)

    prep = prepare_features_only(df, cfg)
    Xdf = prep.df_features

    y = df[cfg.target_name].replace([np.inf, -np.inf], np.nan)

    bad_y_nan = y.isna().to_numpy()
    bad_y_nonfinite = (~np.isfinite(y.to_numpy(dtype=float, copy=False)))

    reasons = dict(prep.reasons)
    reasons["nan_in_target"] = bad_y_nan
    reasons["nonfinite_in_target"] = bad_y_nonfinite

    valid = prep.valid_mask & (~bad_y_nan) & (~bad_y_nonfinite)

    if cfg.target_log10:
        nonpositive_target = (y <= 0).to_numpy()
        reasons["nonpositive_target_for_log10"] = nonpositive_target
        valid = valid & (~nonpositive_target)
        y = _log10_series_safe(y)

    if drop_invalid:
        Xdf = Xdf.loc[valid].copy()
        y = y.loc[valid].copy()
        # After drop, validity relative to returned frames is all True
        prep = PrepResult(df_features=Xdf, valid_mask=np.ones(len(Xdf), dtype=bool), reasons=reasons)
        return Xdf, y, prep

    prep = PrepResult(df_features=Xdf, valid_mask=valid, reasons=reasons)
    return Xdf, y, prep


def inverse_target_log10(yhat_log10: np.ndarray) -> np.ndarray:
    return np.power(10.0, yhat_log10)