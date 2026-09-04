#!/usr/bin/env python3
"""
train_model_final_export.py

End-to-end "final model" training + ONNX export with embedded feature metadata.

Assumptions:
- Clean rows by replacing ±inf -> NaN and dropping invalid rows for TRAINING.
- log10-transform selected input variables (VAR_TO_LOG).
- log10-transform the target during training (TARGET_LOG10=True).
- ONNX model expects inputs ALREADY prepared (i.e., log10 applied to VAR_TO_LOG, finite, non-NaN).

Outputs:
- model.onnx (with feature_names, var_to_log, target transform, training min/max, notes, and versions)
- companion tree variance + domain bundle (joblib) for uncertainty estimation and domain flagging
- companion metadata JSON summarizing model configuration, training data stats, and domain methods
- optional: quick parity check vs onnxruntime predictions on a small batch

Domain Flagging Capabilities:
- Range-based: flags samples outside training feature min/max bounds
- Mahalanobis distance: flags samples with high multivariate distance from training distribution
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# ONNX + conversion
import onnx
from onnx import helper
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# For saving tree variance bundle
import joblib

# Optional parity check (recommended)
try:
    import onnxruntime as ort
    HAS_ORT = True
except Exception:
    HAS_ORT = False

# ----------------------------
# CONFIG
# ----------------------------

from model_config import (
    FEATURE_NAMES,
    VAR_TO_LOG,
    RATIO_DEFS,
    TARGET_NAME,
    TARGET_LOG10,
    RF_PARAMS,
)

# Require explicit file paths at runtime; no local default paths are used.
TRAIN_CSV = Path("")
OUTPUT_ONNX = Path("")
OUTPUT_VARIANCE_BUNDLE = None


def normalize_output_model_path(output_arg: str | Path, model_name: str | None = None) -> Path:
    """Return a concrete .onnx file path from either a file path or a directory path."""
    candidate = Path(output_arg).expanduser()

    # Treat a directory-like path as a directory and use the CLI-supplied model name.
    if candidate.name == "" or str(output_arg).endswith("/") or str(output_arg).endswith("\\") or (candidate.exists() and candidate.is_dir()):
        if not model_name:
            raise ValueError("--model-name is required when --output-model is a directory")
        candidate = candidate / model_name

    # Ensure a file suffix is present and end with .onnx for a predictable bundle naming scheme.
    if candidate.suffix.lower() != ".onnx":
        candidate = candidate.with_suffix(".onnx")

    return candidate.resolve()


def parse_args(argv=None):
    """Parse command-line arguments for final model export."""
    parser = argparse.ArgumentParser(
        description="Train the final diatom biomass model and export it to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--training-data",
        required=True,
        help="Path to the training CSV file used to fit the final model.",
    )
    parser.add_argument(
        "--output-model",
        required=True,
        help="Output ONNX path for the final trained model. A directory is also accepted when --model-name is provided.",
    )
    parser.add_argument(
        "--model-name",
        help="Filename to use when --output-model is a directory. The .onnx suffix is added if needed.",
    )
    return parser.parse_args(argv)


# Domain feature flag configuration
MAHAL_PERCENTILE = 95.0  # Percentile for Mahalanobis threshold

# Freeform notes to embed in ONNX metadata
NOTES = (
    "Final RF model trained after cruise-aware CV. "
    "Inputs listed in feature_names; vars in var_to_log were log10-transformed prior to training/inference. "
    "Target was log10-transformed during training. "
    "Rows with NaN/±inf or non-positive values in log10 vars were excluded from training."
)

# ----------------------------
# Feature preparation utilities
# ----------------------------

# Using prepare_training_xy from prepare_features module


# ----------------------------
# ONNX export utilities
# ----------------------------

def _fmt_float(x: float) -> str:
    return f"{float(x):.10g}"


def compute_mahalanobis_reference(X: np.ndarray, percentile: float = 95.0) -> dict:
    """
    Compute reference parameters for Mahalanobis distance domain flagging.
    
    Parameters:
    -----------
    X : np.ndarray
        Training feature matrix (n_samples, n_features)
    percentile : float
        Percentile threshold for domain flagging (default: 95.0)
    
    Returns:
    --------
    dict with keys:
        - mean: feature means from training data
        - cov_inv: inverse covariance matrix
        - threshold: Mahalanobis distance threshold at given percentile
        - percentile: the percentile used
    """
    # Compute mean and covariance
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    
    # Add small regularization to ensure invertibility
    reg = 1e-6 * np.eye(cov.shape[0])
    cov_reg = cov + reg
    
    # Compute inverse covariance
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if singular
        cov_inv = np.linalg.pinv(cov_reg)
    
    # Compute Mahalanobis distances for all training samples
    diff = X - mean
    mahal_dists = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    
    # Get threshold at specified percentile
    threshold = np.percentile(mahal_dists, percentile)
    
    return {
        "mean": mean,
        "cov_inv": cov_inv,
        "threshold": threshold,
        "percentile": percentile,
    }


def export_rf_to_onnx_with_metadata(
    model: RandomForestRegressor,
    *,
    feature_names: List[str],
    var_to_log: List[str],
    target_name: str,
    target_log10: bool,
    training_min: np.ndarray,
    training_max: np.ndarray,
    onnx_path: Path,
    input_name: str = "X",
    opset: Optional[int] = None,
    notes: Optional[str] = None,
) -> onnx.ModelProto:
    """Convert sklearn model to ONNX and embed metadata."""
    n_features = len(feature_names)
    initial_types = [(input_name, FloatTensorType([None, n_features]))]

    kwargs: Dict[str, Any] = {}
    if opset is not None:
        kwargs["target_opset"] = opset

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        options={},  # regressor: usually fine empty
        **kwargs,
    )

    metadata: Dict[str, str] = {
        "feature_names": ",".join(feature_names),
        "var_to_log": ",".join(var_to_log),
        "n_features": str(n_features),
        "target_name": target_name,
        "target_log10": "true" if target_log10 else "false",
        "training_min": ",".join(_fmt_float(x) for x in training_min.tolist()),
        "training_max": ",".join(_fmt_float(x) for x in training_max.tolist()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    metadata["rf_params"] = json.dumps(RF_PARAMS, sort_keys=True)
    metadata["ratio_defs"] = json.dumps(RATIO_DEFS, sort_keys=True)

    # Version stamps (best-effort)
    try:
        import sklearn
        metadata["sklearn_version"] = sklearn.__version__
    except Exception:
        pass
    try:
        import skl2onnx
        metadata["skl2onnx_version"] = skl2onnx.__version__
    except Exception:
        pass
    try:
        metadata["onnx_version"] = onnx.__version__
    except Exception:
        pass

    if notes:
        metadata["notes"] = notes

    # Attach metadata (overwrite keys if present)
    existing_keys = {p.key for p in onnx_model.metadata_props}
    for k, v in metadata.items():
        if k in existing_keys:
            onnx_model.metadata_props[:] = [p for p in onnx_model.metadata_props if p.key != k]
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = v
        onnx_model.metadata_props.extend([entry])

    onnx.save(onnx_model, str(onnx_path))
    return onnx_model


def read_onnx_metadata(onnx_path: Path) -> Dict[str, str]:
    model = onnx.load(str(onnx_path))
    return {p.key: p.value for p in model.metadata_props}


# ----------------------------
# Optional parity check
# ----------------------------

def onnx_predict(onnx_path: Path, X: np.ndarray, input_name: str = "X") -> np.ndarray:
    """Run ONNX model with onnxruntime; returns yhat array."""
    if not HAS_ORT:
        raise RuntimeError("onnxruntime not installed; pip/conda install onnxruntime")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = sess.run(None, {input_name: X.astype(np.float32, copy=False)})
    # Most sklearn regressors export single output array as first output
    yhat = outputs[0]
    return np.asarray(yhat).reshape(-1)


def inverse_target_log10(yhat_log10: np.ndarray) -> np.ndarray:
    return np.power(10.0, yhat_log10)


# ----------------------------
# Main
# ----------------------------

def main(argv=None) -> None:
    args = parse_args(argv)

    global TRAIN_CSV, OUTPUT_ONNX, OUTPUT_VARIANCE_BUNDLE
    TRAIN_CSV = Path(args.training_data).expanduser().resolve()
    OUTPUT_ONNX = normalize_output_model_path(args.output_model, args.model_name)
    OUTPUT_VARIANCE_BUNDLE = OUTPUT_ONNX.with_name(
        OUTPUT_ONNX.stem + "_tree_variance_domain_bundle.joblib"
    )

    print(f"Loading training data: {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)

    from prepare_features import PrepConfig, prepare_training_xy

    print("Preparing training features/target (drop invalid rows)...")

    cfg = PrepConfig(
        feature_names=FEATURE_NAMES,
        var_to_log=VAR_TO_LOG,
        target_name=TARGET_NAME,
        target_log10=TARGET_LOG10,
        ratio_defs=RATIO_DEFS,
    )

    Xdf, y, prep = prepare_training_xy(df, cfg, drop_invalid=True)

    # prep is a PrepResult with df_features, not Xdf
    # Xdf and y are already returned correctly from prepare_training_xy
    
    assert list(Xdf.columns) == FEATURE_NAMES

    print(f"Training rows: {len(Xdf)} / {len(df)}")
    print("Dropped counts by reason:")
    # prep.reasons contains the drop reasons
    dropped_counts = {k: int(v.sum()) for k, v in prep.reasons.items()}
    print(json.dumps(dropped_counts, indent=2))

    print("Fitting final model...")
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(Xdf.to_numpy(np.float32, copy=False), y.to_numpy(np.float32, copy=False))

    print(f"Exporting ONNX to: {OUTPUT_ONNX}")
    
    # Ensure output directory exists
    OUTPUT_ONNX.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute training min/max from the prepared data
    X_arr = Xdf.to_numpy(dtype=np.float32, copy=False)
    training_min = np.nanmin(X_arr, axis=0)
    training_max = np.nanmax(X_arr, axis=0)
    
    export_rf_to_onnx_with_metadata(
        model=model,
        feature_names=FEATURE_NAMES,
        var_to_log=VAR_TO_LOG,
        target_name=TARGET_NAME,
        target_log10=TARGET_LOG10,
        training_min=training_min,
        training_max=training_max,
        onnx_path=OUTPUT_ONNX,
        input_name="X",
        opset=None,
        notes=NOTES,
    )

    md = read_onnx_metadata(OUTPUT_ONNX)
    print("Wrote ONNX with metadata keys:", sorted(md.keys()))

    # ---- Domain reference quantities from training features ----
    print("Computing domain reference quantities...")
    X64 = Xdf.to_numpy(dtype=np.float64, copy=False)
    feature_min = X64.min(axis=0)
    feature_max = X64.max(axis=0)
    mahal_ref = compute_mahalanobis_reference(X64, percentile=MAHAL_PERCENTILE)
    
    # ---- Save companion bundle for tree-based variance + domain flagging ----
    print(f"Saving tree variance + domain bundle to: {OUTPUT_VARIANCE_BUNDLE}")
    
    # Bundle includes both uncertainty estimation and domain flagging capabilities
    bundle = {
        "feature_names": FEATURE_NAMES,
        "log10_target": TARGET_LOG10,
        "rf_params": RF_PARAMS,
        "estimators": model.estimators_,
        "n_estimators": len(model.estimators_),
        # Domain reference quantities
        "feature_min": feature_min,
        "feature_max": feature_max,
        "mahal_mean": mahal_ref["mean"],
        "mahal_cov_inv": mahal_ref["cov_inv"],
        "mahal_threshold": mahal_ref["threshold"],
        "mahal_percentile": mahal_ref["percentile"],
    }
    joblib.dump(bundle, OUTPUT_VARIANCE_BUNDLE)

    # ---- Save metadata JSON ----
    metadata_json = {
        "feature_names": FEATURE_NAMES,
        "target_name": TARGET_NAME,
        "log10_target": TARGET_LOG10,
        "rf_params": RF_PARAMS,
        "n_estimators": len(model.estimators_),
        "onnx_model": OUTPUT_ONNX.name,
        "tree_variance_bundle": OUTPUT_VARIANCE_BUNDLE.name,
        "training_rows": len(Xdf),
        "total_rows": len(df),
        "var_to_log": VAR_TO_LOG,
        "ratio_defs": RATIO_DEFS,
        # Domain flagging capabilities
        "domain_flag_methods": ["range", "mahalanobis"],
        "mahal_percentile": MAHAL_PERCENTILE,
    }
    
    metadata_path = OUTPUT_ONNX.with_name(OUTPUT_ONNX.stem + "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata_json, f, indent=2)
    
    print(f"Saved tree variance + domain bundle to: {OUTPUT_VARIANCE_BUNDLE}")
    print(f"Saved metadata to: {metadata_path}")

    # Optional parity check
    if HAS_ORT:
        print("Running sklearn vs ONNX parity check on a small batch...")
        n = min(25, len(Xdf))
        X_small = Xdf.iloc[:n].to_numpy(np.float32, copy=False)

        yhat_skl = model.predict(X_small).reshape(-1)
        yhat_onnx = onnx_predict(OUTPUT_ONNX, X_small, input_name="X").reshape(-1)

        max_abs = float(np.max(np.abs(yhat_skl - yhat_onnx)))
        rmse = float(np.sqrt(np.mean((yhat_skl - yhat_onnx) ** 2)))
        print(f"Parity check (in trained space): max_abs={max_abs:.6g}, rmse={rmse:.6g}")

        if TARGET_LOG10:
            # Show example in original space too
            yhat_skl_orig = inverse_target_log10(yhat_skl)
            yhat_onnx_orig = inverse_target_log10(yhat_onnx)
            max_abs_orig = float(np.max(np.abs(yhat_skl_orig - yhat_onnx_orig)))
            rmse_orig = float(np.sqrt(np.mean((yhat_skl_orig - yhat_onnx_orig) ** 2)))
            print(f"Parity check (original space): max_abs={max_abs_orig:.6g}, rmse={rmse_orig:.6g}")
    else:
        print("onnxruntime not installed; skipping parity check.")

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
