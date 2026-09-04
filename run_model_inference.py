#!/usr/bin/env python3
"""
run_model_inference.py

Inference script for the diatom carbon model using the trained ONNX model.

Usage:
    python run_model_inference.py input.csv output.csv model.onnx
    
The script:
1. Loads the ONNX model and reads its metadata
2. Prepares input features according to model requirements
3. Runs inference on valid samples
4. Estimates prediction uncertainty using tree variance from companion Random Forest bundle
5. Performs domain flagging to identify out-of-distribution samples
6. Returns predictions with uncertainty estimates and domain flags

OUTPUTS INCLUDE:
================

Core Predictions:
    - diatCarb_predicted: Primary diatom carbon prediction

Uncertainty Estimates:
    - tree_std_log, tree_var_log: Uncertainty in log space
    - tree_q05_log, tree_q50_log, tree_q95_log: Quantiles in log space
    - tree_std_orig_approx, tree_std: Standard deviation (depends on log10_target)
    - tree_q05, tree_q50, tree_q95: Quantiles in original space

Domain Flags (when available):
    - domain_flag_range: Binary flag for range-based out-of-domain detection
    - domain_flag_mahal: Binary flag for Mahalanobis-based out-of-domain detection
    - domain_flag_any: Binary flag if any domain check fails
    - mahal_dist: Mahalanobis distance from training distribution
    - n_features_outside_range: Count of features outside training range
    - outside_features: Names of features that are out of range (semicolon-separated)

INPUT CSV REQUIREMENTS:
========================

Required Columns (4 base columns):
    - chla: Chlorophyll-a concentration
    - chlb: Chlorophyll-b concentration  
    - chlc: Chlorophyll-c concentration
    - ppc:  Photo-protective carotenoids concentration

Derived Features (automatically calculated):
    - chlb_a = chlb / chla (chlorophyll-b to chlorophyll-a ratio)
    - chlc_a = chlc / chla (chlorophyll-c to chlorophyll-a ratio)
    - ppc_a  = ppc / chla  (photo-protective carotenoids to chlorophyll-a ratio)

Data Quality Requirements:
    - All values must be positive numbers (> 0) for successful predictions
    - Zero, negative, NaN, or infinite values will result in NaN predictions
    - Division by zero when computing ratios will result in NaN predictions
    - Model uses log10 transformations, requiring positive values

CSV Format Example:
    chla,chlb,chlc,ppc
    1.0,0.2,0.3,0.1
    2.0,0.4,0.6,0.2
    0.5,0.1,0.15,0.05

Invalid Data Handling:
    - Samples with invalid data receive NaN predictions
    - Valid samples in the same file will still receive predictions
    - No error thrown for invalid samples, just NaN returned

Units:
    - Input data should use the same units as the training dataset
    - Typically, chlorophyll and carotenoid concentrations are in mg/m^3
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import onnx
import joblib

# ONNX runtime for inference
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("Warning: onnxruntime not installed. Install with: pip install onnxruntime")

import os

from model_config import FEATURE_NAMES, VAR_TO_LOG, RATIO_DEFS, TARGET_NAME, TARGET_LOG10
from prepare_features import PrepConfig, prepare_features_only

MODEL_PATH_ENV_VAR = "DIATOM_INFERENCE_MODEL"


def read_onnx_metadata(onnx_path: Path) -> dict[str, str]:
    """Read metadata from ONNX model."""
    model = onnx.load(str(onnx_path))
    return {p.key: p.value for p in model.metadata_props}


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
    """Convert log10 predictions back to original scale."""
    return np.power(10.0, yhat_log10)


def load_metadata(model_dir: str | Path) -> dict:
    """Load metadata from model directory."""
    model_dir = Path(model_dir)
    metadata_files = [
        "metadata.json", 
    ]
    if model_dir.is_file():
        metadata_files.insert(0, f"{model_dir.stem}_metadata.json")
    
    for metadata_file in metadata_files:
        metadata_path = model_dir / metadata_file if model_dir.is_dir() else model_dir.parent / metadata_file
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
    
    # Fallback to ONNX metadata if no JSON found
    if model_dir.is_file() and model_dir.suffix == ".onnx":
        return read_onnx_metadata(model_dir)
    elif model_dir.is_dir():
        onnx_files = list(model_dir.glob("*.onnx"))
        if len(onnx_files) == 1:
            return read_onnx_metadata(onnx_files[0])
    
    raise FileNotFoundError(f"No metadata found for model at {model_dir}")


def run_onnx_predict(onnx_path: str | Path, X: np.ndarray) -> np.ndarray:
    """Run ONNX model prediction."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    pred = sess.run(None, {input_name: X.astype(np.float32)})[0]
    pred = np.asarray(pred).squeeze()
    return pred


def get_tree_predictions(estimators: list, X: np.ndarray) -> np.ndarray:
    """
    Returns array of shape (n_samples, n_trees)
    """
    per_tree = [est.predict(X).reshape(-1, 1) for est in estimators]
    return np.hstack(per_tree)


def propagate_std_from_log10(y_log: np.ndarray, std_log: np.ndarray) -> np.ndarray:
    """
    Approximate standard deviation in original space using first-order propagation:
        y = 10^z
        sigma_y ≈ ln(10) * 10^z * sigma_z
    """
    y_orig = np.power(10.0, y_log)
    return np.log(10.0) * y_orig * std_log


def mahalanobis_distance(X: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    """
    Compute Mahalanobis distance for each sample.
    
    Parameters:
    -----------
    X : np.ndarray
        Input samples (n_samples, n_features)
    mean : np.ndarray
        Reference mean vector (n_features,)
    cov_inv : np.ndarray
        Inverse covariance matrix (n_features, n_features)
    
    Returns:
    --------
    distances : np.ndarray
        Mahalanobis distances (n_samples,)
    """
    diffs = X - mean
    d2 = np.einsum("ij,jk,ik->i", diffs, cov_inv, diffs)
    return np.sqrt(np.maximum(d2, 0.0))


def compute_range_flags(X: np.ndarray, feature_min: np.ndarray, feature_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute range-based domain flags.
    
    Parameters:
    -----------
    X : np.ndarray
        Input samples (n_samples, n_features)
    feature_min : np.ndarray
        Minimum training values per feature (n_features,)
    feature_max : np.ndarray
        Maximum training values per feature (n_features,)
    
    Returns:
    --------
    outside_mask : np.ndarray
        Boolean mask of shape (n_samples, n_features) indicating which features are outside range
    any_outside : np.ndarray
        Boolean array of shape (n_samples,) indicating if any feature is outside range
    """
    below = X < feature_min
    above = X > feature_max
    outside_mask = below | above
    any_outside = outside_mask.any(axis=1)
    return outside_mask, any_outside


def load_model_and_config(model_path: Path) -> tuple[dict, PrepConfig]:
    """Load ONNX model metadata and create PrepConfig."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    metadata = read_onnx_metadata(model_path)
    
    # Extract configuration from metadata
    feature_names = FEATURE_NAMES
    var_to_log = VAR_TO_LOG
    target_name = metadata.get("target_name", TARGET_NAME)
    target_log10 = metadata.get("target_log10", "false").lower() == "true"
    
    # Parse ratio definitions if present
    ratio_defs = RATIO_DEFS
    
    cfg = PrepConfig(
        feature_names=feature_names,
        var_to_log=var_to_log,
        target_name=target_name,
        target_log10=target_log10,
        ratio_defs=ratio_defs,
    )
    
    return metadata, cfg


def predict_diatom_carbon_with_uncertainty(
    df_input: pd.DataFrame,
    model_path: Path,
    return_metadata: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Predict diatom carbon concentrations with uncertainty estimates.
    
    Parameters:
    -----------
    df_input : pd.DataFrame
        Input dataframe with required columns
    model_path : Path
        Path to ONNX model file or model directory
    return_metadata : bool
        Whether to return model metadata along with predictions
    
    Returns:
    --------
    df_output : pd.DataFrame
        Dataframe with predictions and uncertainty estimates
    metadata : dict (optional)
        Model metadata if return_metadata=True
    """
    # Determine model directory and files
    if model_path.is_dir():
        model_dir = model_path
        onnx_files = list(model_dir.glob("*.onnx"))
        if len(onnx_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one ONNX model in {model_dir}, found {len(onnx_files)}"
            )
        onnx_file = onnx_files[0]
        bundle_file = model_dir / f"{onnx_file.stem}_tree_variance_domain_bundle.joblib"
    else:
        model_dir = model_path.parent
        onnx_file = model_path
        # Try to find the domain bundle file
        bundle_file = model_dir / f"{model_path.stem}_tree_variance_domain_bundle.joblib"
        # Fallback to regular variance bundle
        if not bundle_file.exists():
            bundle_file = model_dir / f"{model_path.stem}_tree_variance_bundle.joblib"
    
    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_file}")
    if not bundle_file.exists():
        raise FileNotFoundError(f"Tree variance bundle not found: {bundle_file}")
    
    # Load metadata and configuration
    metadata = load_metadata(model_dir)
    
    feature_names = FEATURE_NAMES
    var_to_log = VAR_TO_LOG
    
    target_name = metadata.get("target_name", TARGET_NAME)
    log10_target = bool(metadata.get("log10_target", metadata.get("target_log10", TARGET_LOG10)))
    
    # Check if domain flagging is available
    domain_methods = metadata.get("domain_flag_methods", [])
    has_domain_flags = len(domain_methods) > 0
    
    # Parse ratio definitions if present
    ratio_defs = RATIO_DEFS
    
    cfg = PrepConfig(
        feature_names=feature_names,
        var_to_log=var_to_log,
        target_name=target_name,
        target_log10=log10_target,
        ratio_defs=ratio_defs,
    )
    
    print(f"Loaded model: {onnx_file}")
    print(f"Loaded bundle: {bundle_file}")
    print(f"Model expects {len(cfg.feature_names)} features: {cfg.feature_names}")
    print(f"Variables to log10-transform: {cfg.var_to_log}")
    if has_domain_flags:
        print(f"Domain flagging methods available: {domain_methods}")
    
    # Clean input data
    df = df_input.replace([np.inf, -np.inf], np.nan).copy()
    
    # Prepare features using the existing pipeline FIRST
    prep = prepare_features_only(df, cfg)
    
    # Create output dataframe
    pred_df = df.copy()
    valid_mask = prep.valid_mask
    
    # Initialize output columns
    pred_df["diatCarb_predicted"] = np.nan
    pred_df["tree_std_log"] = np.nan
    pred_df["tree_var_log"] = np.nan
    pred_df["tree_q05_log"] = np.nan
    pred_df["tree_q50_log"] = np.nan
    pred_df["tree_q95_log"] = np.nan
    
    # Initialize domain flag columns if available
    if has_domain_flags:
        pred_df["mahal_dist"] = np.nan
        pred_df["domain_flag_range"] = np.nan
        pred_df["domain_flag_mahal"] = np.nan
        pred_df["domain_flag_any"] = np.nan
        pred_df["n_features_outside_range"] = np.nan
        pred_df["outside_features"] = ""
    
    if log10_target:
        pred_df["tree_std_orig_approx"] = np.nan
        pred_df["tree_q05"] = np.nan
        pred_df["tree_q50"] = np.nan
        pred_df["tree_q95"] = np.nan
    else:
        pred_df["tree_std"] = np.nan
        pred_df["tree_q05"] = np.nan
        pred_df["tree_q50"] = np.nan
        pred_df["tree_q95"] = np.nan
    
    print(f"Input samples: {len(df)}")
    print(f"Valid samples: {valid_mask.sum()}")
    
    if valid_mask.sum() == 0:
        print("No valid rows to predict.")
        if return_metadata:
            return pred_df, metadata
        else:
            return pred_df
    
    # Use the prepared features
    X = prep.df_features.loc[valid_mask].astype(np.float32).to_numpy()
    
    # ---- Mean prediction from ONNX ----
    y_pred_onnx = run_onnx_predict(onnx_file, X)
    
    # ---- Per-tree predictions from companion bundle ----
    bundle = joblib.load(bundle_file)
    estimators = bundle["estimators"]
    
    tree_preds = get_tree_predictions(estimators, X)  # (n_samples, n_trees)
    
    # Tree-based spread in model space
    tree_mean = np.mean(tree_preds, axis=1)
    tree_std = np.std(tree_preds, axis=1, ddof=1)
    tree_var = np.var(tree_preds, axis=1, ddof=1)
    tree_q05 = np.quantile(tree_preds, 0.05, axis=1)
    tree_q50 = np.quantile(tree_preds, 0.50, axis=1)
    tree_q95 = np.quantile(tree_preds, 0.95, axis=1)
    
    # ---- Domain flags (if available) ----
    if has_domain_flags:
        # Use float64 for domain calculations for better precision
        X64 = X.astype(np.float64)
        
        # Range-based flags
        if "feature_min" in bundle and "feature_max" in bundle:
            feature_min = np.asarray(bundle["feature_min"], dtype=np.float64)
            feature_max = np.asarray(bundle["feature_max"], dtype=np.float64)
            outside_mask, any_outside = compute_range_flags(X64, feature_min, feature_max)
            n_outside = outside_mask.sum(axis=1)
            
            # Create list of feature names that are outside range
            outside_feature_names = []
            for row_mask in outside_mask:
                names = [feature_names[i] for i, flag in enumerate(row_mask) if flag]
                outside_feature_names.append(";".join(names))
        else:
            any_outside = np.zeros(len(X64), dtype=bool)
            n_outside = np.zeros(len(X64), dtype=int)
            outside_feature_names = [""] * len(X64)
        
        # Mahalanobis-based flags
        if all(k in bundle for k in ["mahal_mean", "mahal_cov_inv", "mahal_threshold"]):
            mahal_mean = np.asarray(bundle["mahal_mean"], dtype=np.float64)
            mahal_cov_inv = np.asarray(bundle["mahal_cov_inv"], dtype=np.float64)
            mahal_threshold = float(bundle["mahal_threshold"])
            
            mahal_dist = mahalanobis_distance(X64, mahal_mean, mahal_cov_inv)
            mahal_flag = mahal_dist > mahal_threshold
        else:
            mahal_dist = np.zeros(len(X64))
            mahal_flag = np.zeros(len(X64), dtype=bool)
        
        # Combined domain flag
        domain_flag_any = any_outside | mahal_flag
    
    # Store log-space uncertainties
    pred_df.loc[valid_mask, "tree_std_log"] = tree_std
    pred_df.loc[valid_mask, "tree_var_log"] = tree_var
    pred_df.loc[valid_mask, "tree_q05_log"] = tree_q05
    pred_df.loc[valid_mask, "tree_q50_log"] = tree_q50
    pred_df.loc[valid_mask, "tree_q95_log"] = tree_q95
    
    # Store domain flags if available
    if has_domain_flags:
        pred_df.loc[valid_mask, "mahal_dist"] = mahal_dist
        pred_df.loc[valid_mask, "domain_flag_range"] = any_outside.astype(int)
        pred_df.loc[valid_mask, "domain_flag_mahal"] = mahal_flag.astype(int)
        pred_df.loc[valid_mask, "domain_flag_any"] = domain_flag_any.astype(int)
        pred_df.loc[valid_mask, "n_features_outside_range"] = n_outside
        pred_df.loc[valid_mask, "outside_features"] = outside_feature_names
    
    if log10_target:
        # Convert prediction and quantiles back to original space
        pred_df.loc[valid_mask, "diatCarb_predicted"] = np.power(10.0, y_pred_onnx)
        pred_df.loc[valid_mask, "tree_q05"] = np.power(10.0, tree_q05)
        pred_df.loc[valid_mask, "tree_q50"] = np.power(10.0, tree_q50)
        pred_df.loc[valid_mask, "tree_q95"] = np.power(10.0, tree_q95)
        pred_df.loc[valid_mask, "tree_std_orig_approx"] = propagate_std_from_log10(
            y_pred_onnx, tree_std
        )
    else:
        # Use ONNX prediction as primary prediction
        pred_df.loc[valid_mask, "diatCarb_predicted"] = y_pred_onnx
        pred_df.loc[valid_mask, "tree_std"] = tree_std
        pred_df.loc[valid_mask, "tree_q05"] = tree_q05
        pred_df.loc[valid_mask, "tree_q50"] = tree_q50
        pred_df.loc[valid_mask, "tree_q95"] = tree_q95
    
    if return_metadata:
        return pred_df, metadata
    else:
        return pred_df


def main():
    """Command-line interface for batch prediction with uncertainty."""
    if len(sys.argv) < 2:
        print("Usage: python run_model_inference.py input.csv [output.csv] model.onnx")
        print("  input.csv  : Input CSV file with environmental data")
        print("  output.csv : Output CSV file (optional, defaults to input_predictions.csv)")
        print("  model.onnx : ONNX model file or directory")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_path.parent / f"{input_path.stem}_predictions.csv"
    
    if len(sys.argv) <= 3:
        model_path_value = os.getenv(MODEL_PATH_ENV_VAR)
        if not model_path_value:
            print(f"Error: model path is required as an argument or via ${MODEL_PATH_ENV_VAR}", file=sys.stderr)
            sys.exit(1)
        model_path = Path(model_path_value)
    else:
        model_path = Path(sys.argv[3])
    
    print(f"Loading data from: {input_path}")
    df_input = pd.read_csv(input_path)
    
    # Make predictions with uncertainty
    df_output, metadata = predict_diatom_carbon_with_uncertainty(df_input, model_path, return_metadata=True)
    
    # Save results
    df_output.to_csv(output_path, index=False)
    print(f"Predictions with uncertainties saved to: {output_path}")
    
    # Print summary
    valid_predictions = ~np.isnan(df_output["diatCarb_predicted"])
    print(f"\nSummary:")
    print(f"  Total samples: {len(df_output)}")
    print(f"  Valid predictions: {valid_predictions.sum()}")
    print(f"  Invalid samples: {(~valid_predictions).sum()}")
    
    if valid_predictions.sum() > 0:
        predictions = df_output["diatCarb_predicted"].values
        print(f"  Prediction range: {np.nanmin(predictions):.3f} - {np.nanmax(predictions):.3f}")
        print(f"  Mean prediction: {np.nanmean(predictions):.3f}")
        
        # Show uncertainty summary
        log10_target = bool(metadata.get("log10_target", metadata.get("target_log10", TARGET_LOG10)))
        
        if log10_target:
            uncertainties = df_output["tree_std_orig_approx"].values
            print(f"  Mean uncertainty (std): {np.nanmean(uncertainties):.3f}")
            print(f"  Uncertainty range: {np.nanmin(uncertainties):.3f} - {np.nanmax(uncertainties):.3f}")
        else:
            uncertainties = df_output["tree_std"].values
            print(f"  Mean uncertainty (std): {np.nanmean(uncertainties):.3f}")
            print(f"  Uncertainty range: {np.nanmin(uncertainties):.3f} - {np.nanmax(uncertainties):.3f}")
        
        # Show domain flag summary if available
        if "domain_flag_any" in df_output.columns:
            domain_flags = df_output["domain_flag_any"].values
            n_flagged = np.nansum(domain_flags[valid_predictions])
            print(f"  Samples flagged as out-of-domain: {n_flagged} ({100*n_flagged/valid_predictions.sum():.1f}%)")
            
            if "domain_flag_range" in df_output.columns:
                range_flags = df_output["domain_flag_range"].values
                n_range_flagged = np.nansum(range_flags[valid_predictions])
                print(f"    Range-based flags: {n_range_flagged}")
            
            if "domain_flag_mahal" in df_output.columns:
                mahal_flags = df_output["domain_flag_mahal"].values
                n_mahal_flagged = np.nansum(mahal_flags[valid_predictions])
                print(f"    Mahalanobis-based flags: {n_mahal_flagged}")


if __name__ == "__main__":
    main()
