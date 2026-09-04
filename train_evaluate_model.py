#!/usr/bin/env python3
"""
train_evaluate_model.py

Train/validate and test the diatom biomass random forest model using the shared
feature definitions and preprocessing defined in model_config.py and
prepare_features.py.

Strata for defining chlorophyll a bins are defined as:
'0.001-0.03', '0.03-0.17', '0.17-1', '1-5', '5-30' mg/m^3

See README.md for the full usage instructions and examples.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from collections import defaultdict

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score, GroupShuffleSplit, cross_val_predict, train_test_split, learning_curve
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.inspection import permutation_importance

# Import forestci for uncertainty quantification
import forestci as fci

# Import model configuration and feature preparation utilities
from model_config import (
    FEATURE_NAMES,
    RATIO_DEFS,
    VAR_TO_LOG,
    TARGET_NAME,
    TARGET_LOG10,
    MODEL_TYPE,
    RF_PARAMS,
    CRUISE_MANUAL_COLORS,
    CRUISE_NAME_MAP,
)
from prepare_features import PrepConfig, prepare_training_xy, compute_derived_features

# Configuration is set from explicit CLI arguments or environment overrides.
DATA_PATH = os.getenv("DIATOM_TRAINING_DATA")
FIGURES_DIR = os.getenv("DIATOM_FIGURES_DIR", "figures")


def parse_args(argv=None):
    """Parse command-line arguments for training/data configuration."""
    parser = argparse.ArgumentParser(
        description="Train and validate the diatom biomass random forest model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--training-data",
        required=True,
        help="Path to the training CSV file (required).",
    )
    parser.add_argument(
        "--figures-dir",
        default=FIGURES_DIR,
        help="Directory for validation and test plots and summary CSV outputs.",
    )
    return parser.parse_args(argv)

def get_cruise_color_mapping(train_cruises, test_cruises, manual_colors=None):
    """
    Create a consistent color mapping for cruises across all plots.
    
    Parameters:
    -----------
    train_cruises : list or array-like
        List of training cruise names
    test_cruises : list or array-like  
        List of hold-out test cruise names
    manual_colors : list, optional
        List of hexadecimal color codes (e.g., ['#FF5733', '#33C7FF', '#8E44AD']).
        If None (default), uses CRUISE_MANUAL_COLORS from model_config.py.
        Colors are assigned sequentially to cruises in sorted order.
        If there are more cruises than manual colors, remaining cruises use automatic tab20 colors.
    
    Returns:
    --------
    dict : Dictionary mapping cruise names to color values (either hex strings or rgba tuples)
    """
    import matplotlib.colors as mcolors
    
    # Use configuration from model_config.py if no manual colors provided
    if manual_colors is None:
        manual_colors = CRUISE_MANUAL_COLORS
    
    all_cruises = sorted(list(set(train_cruises) | set(test_cruises)))
    cmap = plt.colormaps['tab20']
    
    cruise_colors = {}
    
    # Validate manual colors if provided
    if manual_colors:
        for i, color_code in enumerate(manual_colors):
            if not isinstance(color_code, str) or not color_code.startswith('#'):
                raise ValueError(f"Invalid hex color '{color_code}' at position {i}. "
                               "Colors must be hex strings starting with '#' (e.g., '#FF5733')")
            try:
                # Validate that it's a proper hex color
                mcolors.to_rgba(color_code)
            except ValueError:
                raise ValueError(f"Invalid hex color '{color_code}' at position {i}. "
                               "Please use valid hex format (e.g., '#FF5733' or '#FF5733FF')")
    
    # Assign colors to all cruises
    auto_color_index = 0
    for i, cruise in enumerate(all_cruises):
        if manual_colors and i < len(manual_colors):
            # Use manual color from the list
            cruise_colors[cruise] = manual_colors[i]
        else:
            # Use automatic tab20 color
            cruise_colors[cruise] = cmap(auto_color_index / 19.0)  # Normalize to [0,1] range for tab20
            auto_color_index += 1
    
    return cruise_colors


def get_cruise_label(cruise, name_map=None):
    """
    Return a user-friendly label for a cruise. If a name_map is provided and contains
    the cruise key, return the mapped name. Otherwise fall back to 'Dataset {cruise}'.
    """
    if name_map is None:
        name_map = CRUISE_NAME_MAP
    try:
        # Try direct lookup (keys may be int or string)
        if name_map and cruise in name_map:
            return name_map[cruise]
        # try with int key
        if name_map and int(cruise) in name_map:
            return name_map[int(cruise)]
    except Exception:
        pass
    # Default label
    try:
        return f"Dataset {int(cruise)}"
    except Exception:
        return f"Dataset {cruise}"

def create_model(model_type=None):
    """
    Function to create the Random Forest model.
    """
    if model_type is None:
        model_type = MODEL_TYPE
    
    if model_type.lower() == 'rf':
        return RandomForestRegressor(**RF_PARAMS)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Only 'rf' (Random Forest) is supported")

def get_model_params(model_type=None):
    """
    Get parameters for the Random Forest model.
    """
    if model_type is None:
        model_type = MODEL_TYPE
    
    if model_type.lower() == 'rf':
        return RF_PARAMS
    else:
        raise ValueError(f"Unknown model type: {model_type}. Only 'rf' (Random Forest) is supported")

def load_and_prepare_data(data_path):
    """
    Load data and compute derived features using model configuration.
    """
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    
    print("Computing derived features using model_config...")
    cfg = PrepConfig(
        feature_names=FEATURE_NAMES,
        var_to_log=VAR_TO_LOG,
        target_name=TARGET_NAME,
        target_log10=TARGET_LOG10,
        ratio_defs=RATIO_DEFS,
    )
    
    # Compute derived features (ratios)
    data = compute_derived_features(data, cfg)
    
    # Create stratum column based on chlorophyll a bins
    print("Creating stratum column based on chlorophyll a bins...")
    bins = [0.001, 0.03, 0.17, 1, 5, 30]
    bin_labels = ['0.001-0.03', '0.03-0.17', '0.17-1', '1-5', '5-30']
    data['stratum'] = pd.cut(data['chla'], bins=bins, labels=bin_labels, include_lowest=True)
    
    print(f"Stratum distribution:")
    print(data['stratum'].value_counts().sort_index())
    
    return data, cfg

def create_train_test_split(df, cfg, test_size=0.25, random_state=42):
    """
    Create chl_a stratified train/test split by cruise/cruise leg, maintaining stratum distribution of chl_a values
    """
    print("Creating chl_a stratified train/test split by cruise leg and stratum...")
    
    # Identify unique cruises (groups) and assign each cruise to a stratum
    # using its median chlorophyll-a value.
    groups = df['cruise'].dropna().unique()
    cruise_chla = df.groupby('cruise')['chla'].median()
    cruise_strata = pd.cut(
        cruise_chla,
        bins=[0.001, 0.03, 0.17, 1, 5, 30],
        labels=['0.001-0.03', '0.03-0.17', '0.17-1', '1-5', '5-30'],
        include_lowest=True,
    )
    print(f"Total unique cruises: {len(groups)}")
    
    # Create a dictionary to hold group indices for each stratum
    stratum_groups = defaultdict(list)
    for group in groups:
        stratum = cruise_strata[group]
        stratum_groups[stratum].append(group)
    
    print("Cruises per stratum:")
    for stratum, cruise_list in stratum_groups.items():
        print(f"  {stratum}: {len(cruise_list)} cruises - {cruise_list}")
    
    # Split groups in each stratum
    train_cruises = []
    test_cruises = []
    for stratum, cruise_list in stratum_groups.items():
        cruise_list = sorted(cruise_list)
        if len(cruise_list) > 1:
            train_grp, test_grp = train_test_split(
                cruise_list, test_size=test_size, random_state=random_state
            )
            train_cruises.extend(train_grp)
            test_cruises.extend(test_grp)
            print(f"  {stratum}: {len(train_grp)} train, {len(test_grp)} test")
        else:
            train_cruises.extend(cruise_list)
            print(f"  {stratum}: {len(cruise_list)} train (single cruise)")
    
    # Create train/test splits
    train_data = df[df['cruise'].isin(train_cruises)].copy()
    test_data = df[df['cruise'].isin(test_cruises)].copy()
    
    print(f"Training cruises: {sorted(train_cruises)}")
    print(f"Validation cruises: {sorted(test_cruises)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(test_data)}")
    
    # Check for overlap (should be empty)
    overlap = set(train_cruises) & set(test_cruises)
    if overlap:
        print(f"WARNING: Overlap cruises found: {overlap}")
    
    return train_data, test_data, train_cruises, test_cruises

def prepare_datasets(train_data, test_data, cfg):
    """
    Prepare training and test datasets using structured feature preparation.
    """
    print("Preparing training and test datasets...")
    
    # Prepare training data
    X_train, y_train, train_prep_result = prepare_training_xy(train_data, cfg, drop_invalid=True)
    
    # Prepare test data
    X_test, y_test, test_prep_result = prepare_training_xy(test_data, cfg, drop_invalid=True)
    
    # Extract cruise information for valid rows
    # Since drop_invalid=True, we need to get cruise info for the same rows
    train_cruises = train_data.loc[X_train.index, 'cruise'] if 'cruise' in train_data.columns else None
    test_cruises = test_data.loc[X_test.index, 'cruise'] if 'cruise' in test_data.columns else None
    
    print(f"Training data preparation:")
    print(f"  Original: {len(train_data)} rows")
    print(f"  Valid: {len(X_train)} rows")
    
    print(f"Test data preparation:")
    print(f"  Original: {len(test_data)} rows") 
    print(f"  Valid: {len(X_test)} rows")
    
    print(f"Features: {list(X_train.columns)}")
    print(f"Variables log-transformed: {VAR_TO_LOG}")
    print(f"Target log-transformed: {TARGET_LOG10}")
    
    return X_train, y_train, X_test, y_test, train_prep_result, train_cruises, test_cruises

def calc_metrics(y_true_orig, y_pred_orig, y_true_log, y_pred_log, dataset_name):
    """
    Calculate evaluation metrics for model predictions.
    
    Parameters:
    - y_true_orig: actual values in original scale
    - y_pred_orig: predicted values in original scale  
    - y_true_log: actual values in log scale (if applicable)
    - y_pred_log: predicted values in log scale (if applicable)
    - dataset_name: name for printing (e.g., "Training" or "Test")
    """
    # Ensure arrays are numpy arrays and handle any NaN/inf values
    y_true_orig = np.asarray(y_true_orig)
    y_pred_orig = np.asarray(y_pred_orig)
    y_true_log = np.asarray(y_true_log) 
    y_pred_log = np.asarray(y_pred_log)
    
    # Check for NaN or infinite values and their alignment
    valid_mask = (np.isfinite(y_true_orig) & np.isfinite(y_pred_orig) & 
                 np.isfinite(y_true_log) & np.isfinite(y_pred_log))
    
    if not np.all(valid_mask):
        n_invalid = np.sum(~valid_mask)
        print(f"Warning: {n_invalid} invalid values detected in {dataset_name} data, excluding from metrics")
        y_true_orig = y_true_orig[valid_mask]
        y_pred_orig = y_pred_orig[valid_mask] 
        y_true_log = y_true_log[valid_mask]
        y_pred_log = y_pred_log[valid_mask]
    
    # Basic verification
    assert len(y_true_orig) == len(y_pred_orig), f"Length mismatch in {dataset_name}: {len(y_true_orig)} vs {len(y_pred_orig)}"
    print(f"\\n{dataset_name} data summary: {len(y_true_orig)} valid samples")
    print(f"  Actual range (orig): {y_true_orig.min():.4f} to {y_true_orig.max():.4f}")
    print(f"  Predicted range (orig): {y_pred_orig.min():.4f} to {y_pred_orig.max():.4f}")
    
    # Calculate metrics on original scale
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    medae = median_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2 = r2_score(y_true_orig, y_pred_orig)
    
    # Calculate Spearman correlation with explicit debugging
    spearman_rho = pd.Series(y_true_orig).corr(pd.Series(y_pred_orig), method='spearman')
    if pd.isna(spearman_rho):
        print(f"Warning: Spearman correlation returned NaN for {dataset_name}")
        spearman_rho = 0.0
    
    # Calculate R² on log-transformed values if applicable
    if TARGET_LOG10:
        r2_log = r2_score(y_true_log, y_pred_log)
    else:
        r2_log = r2  # Same as regular R² if no log transform
    
    mape = mean_absolute_percentage_error(y_true_orig, y_pred_orig) * 100  # Convert to percentage
    
    # Calculate relative errors with protection against division by zero
    relative_errors = np.abs((y_true_orig - y_pred_orig) / np.maximum(y_true_orig, 1e-10)) * 100
    median_relative_error = np.median(relative_errors)
    
    # Calculate Mean Bias Error (MBE)
    bias = np.mean(y_pred_orig - y_true_orig)
    # Calculate Median Bias Error
    bias_median = np.median(y_pred_orig - y_true_orig)
    # Calculate Mean Percentage Error (MPE)
    perc_errors = (y_true_orig - y_pred_orig) / np.maximum(y_true_orig, 1e-10) * 100
    mpe = np.mean(perc_errors)
    
    print(f"\\n{dataset_name} Metrics (original scale):")
    print(f"  MAE: {mae:.4f}")
    print(f"  MedAE: {medae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    if TARGET_LOG10:
        print(f"  R² (log): {r2_log:.4f}")
    print(f"  MAPE: {mape:.4f}")
    print(f"  Med RelE: {median_relative_error:.4f}")
    print(f"  Mean Bias Error (MBE): {bias:.4f}")
    print(f"  Median Bias Error: {bias_median:.4f}")
    print(f"  MPE: {mpe:.4f}")
    print(f"  Spearman's rho: {spearman_rho:.4f}")
    
    return {
        'mae': mae,
        'medae': medae, 
        'rmse': rmse,
        'r2': r2,
        'r2_log': r2_log,
        'mape': mape,
        'mean_percentage_error': mpe,
        'med_percentage_error': median_relative_error,
        'mean_bias_error': bias,
        'median_bias_error': bias_median,
        'spearman_rho': spearman_rho
    }

def evaluate_model(X_train, y_train, X_test, y_test):
    """
    Train model using config parameters and evaluate performance on a hold-out test set.
    """
    print(f"Training {MODEL_TYPE.upper()} with parameters from model_config:")
    model_params = get_model_params()
    print(json.dumps(model_params, indent=2, default=str))
    
    # Train model using exact config parameters
    model = create_model()
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate uncertainty estimates using forestci
    uncertainty_train = None
    uncertainty_test = None
    
    if MODEL_TYPE.lower() in ['rf', 'randomforest']:
        print("\nCalculating prediction uncertainties using forestci...")
        try:
            # Calculate variance for training and test sets
            variance_train = fci.random_forest_error(model, X_train.values, X_train.values)
            variance_test = fci.random_forest_error(model, X_train.values, X_test.values)
            
            # Store standard deviations (square root of variance)
            uncertainty_train = np.sqrt(variance_train)
            uncertainty_test = np.sqrt(variance_test)
            
            print(f"  Training uncertainty (mean ± std): {uncertainty_train.mean():.4f} ± {uncertainty_train.std():.4f}")
            print(f"  Test uncertainty (mean ± std): {uncertainty_test.mean():.4f} ± {uncertainty_test.std():.4f}")
            
        except Exception as e:
            print(f"Warning: Failed to calculate uncertainties: {e}")
            uncertainty_train = None
            uncertainty_test = None
    
    # Convert back to original scale if log-transformed
    if TARGET_LOG10:
        y_train_orig = np.power(10, y_train)
        y_test_orig = np.power(10, y_test) 
        y_pred_train_orig = np.power(10, y_pred_train)
        y_pred_test_orig = np.power(10, y_pred_test)
    else:
        y_train_orig = y_train
        y_test_orig = y_test
        y_pred_train_orig = y_pred_train
        y_pred_test_orig = y_pred_test
    
    # Calculate metrics using the module-level calc_metrics function
    
    train_metrics = calc_metrics(y_train_orig, y_pred_train_orig, y_train, y_pred_train, "Training")
    test_metrics = calc_metrics(y_test_orig, y_pred_test_orig, y_test, y_pred_test, "Test")
    
    # Store uncertainty estimates in results
    uncertainty_results = {
        'train_uncertainty': uncertainty_train,
        'test_uncertainty': uncertainty_test,
        'has_uncertainty': MODEL_TYPE.lower() in ['rf', 'randomforest']
    }
    
    return model, (y_pred_train_orig, y_pred_test_orig), (train_metrics, test_metrics), uncertainty_results

def plot_predictions(y_train, y_pred_train, y_test, y_pred_test, train_metrics, test_metrics, 
                    train_cruises=None, test_cruises=None, uncertainty_results=None):
    """
    Create prediction vs actual plots on log scale with points colored by cruise.
    If uncertainty_results is provided, creates plots with error bars.
    """
    if uncertainty_results and uncertainty_results['has_uncertainty']:
        # Create plots with error bars
        plot_predictions_with_error_bars(y_train, y_pred_train, y_test, y_pred_test, 
                                        train_metrics, test_metrics, uncertainty_results,
                                        train_cruises, test_cruises)
    
    # Also create regular plots without error bars for comparison
    plot_predictions_standard(y_train, y_pred_train, y_test, y_pred_test, 
                             train_metrics, test_metrics, train_cruises, test_cruises)

def plot_predictions_standard(y_train, y_pred_train, y_test, y_pred_test, train_metrics, test_metrics, 
                            train_cruises=None, test_cruises=None):
    """
    Create standard prediction vs actual plots on log scale with points colored by cruise.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get consistent cruise color mapping
    if train_cruises is not None and test_cruises is not None:
        cruise_colors = get_cruise_color_mapping(train_cruises, test_cruises)
    
    # Training plot
    if train_cruises is not None:
        unique_train_cruises = sorted(train_cruises.unique())
        
        # Plot each cruise with its specific color
        for cruise in unique_train_cruises:
            mask = train_cruises == cruise
            ax1.scatter(y_train[mask], y_pred_train[mask], alpha=0.6, 
                       color=cruise_colors[cruise], s=20, 
                       label=get_cruise_label(cruise))
        
        ax1.legend(title='Training Cruises', loc='upper left', fontsize=8, title_fontsize=9)
    else:
        ax1.scatter(y_train, y_pred_train, alpha=0.6, color='blue', s=20)
    
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    ax1.set_xlabel('In situ measured DiatC (mg m⁻³)')
    ax1.set_ylabel('Predicted DiatC (mg m⁻³)')
    ax1.set_title(f'Training Set  R² = {train_metrics["r2"]:.3f}, RMSE = {train_metrics["rmse"]:.3f}')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Test plot 
    if test_cruises is not None:
        unique_test_cruises = sorted(test_cruises.unique())
        
        # Plot each cruise with its specific color
        for cruise in unique_test_cruises:
            mask = test_cruises == cruise
            ax2.scatter(y_test[mask], y_pred_test[mask], alpha=0.6, 
                       color=cruise_colors[cruise], s=20, 
                       label=get_cruise_label(cruise))
        
        ax2.legend(title='Validation Cruises', loc='upper left', fontsize=8, title_fontsize=9)
    else:
        ax2.scatter(y_test, y_pred_test, alpha=0.6, color='red', s=20)
    
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax2.set_xlabel('In situ measured DiatC (mg m⁻³)')
    ax2.set_ylabel('Predicted DiatC (mg m⁻³)')
    ax2.set_title(f'Test Set  R² = {test_metrics["r2"]:.3f}, RMSE = {test_metrics["rmse"]:.3f}')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save prediction plots
    figures_dir = Path(FIGURES_DIR)
    figures_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(figures_dir / 'predictions_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Standard prediction comparison plots saved to {figures_dir / 'predictions_comparison.png'}")
    plt.close()  # Close to avoid interactive display

def plot_predictions_with_error_bars(y_train, y_pred_train, y_test, y_pred_test, 
                                    train_metrics, test_metrics, uncertainty_results,
                                    train_cruises=None, test_cruises=None):
    """
    Create prediction vs actual plots with error bars showing uncertainty estimates.
    """
    train_uncertainty = uncertainty_results['train_uncertainty']
    test_uncertainty = uncertainty_results['test_uncertainty']
    
    # 27% measurement error on in situ DiatC values (x-axis), based on Chase et al. 2022 (https://doi.org/10.1029/2022GL098076)
    INSITU_ERROR_FRAC = 0.27
    
    # Convert uncertainties to original scale if needed
    if TARGET_LOG10 and train_uncertainty is not None:
        # For log-transformed predictions, uncertainty needs to be propagated
        # σ_y ≈ y_pred * ln(10) * σ_log_y
        conversion_factor = np.log(10)
        train_uncertainty_orig = y_pred_train * conversion_factor * train_uncertainty
        test_uncertainty_orig = y_pred_test * conversion_factor * test_uncertainty
    else:
        train_uncertainty_orig = train_uncertainty
        test_uncertainty_orig = test_uncertainty
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get consistent cruise color mapping
    if train_cruises is not None and test_cruises is not None:
        cruise_colors = get_cruise_color_mapping(train_cruises, test_cruises)
    
    # Training plot with error bars
    if train_cruises is not None:
        unique_train_cruises = sorted(train_cruises.unique())
        
        # Plot each cruise with its specific color and error bars
        for cruise in unique_train_cruises:
            mask = train_cruises == cruise
            if train_uncertainty_orig is not None:
                ax1.errorbar(y_train[mask], y_pred_train[mask], 
                           xerr=y_train[mask] * INSITU_ERROR_FRAC,
                           yerr=train_uncertainty_orig[mask],
                           fmt='o', alpha=0.6, color=cruise_colors[cruise],
                           markersize=4, capsize=None, capthick=0.5,
                           elinewidth=0.5, label=get_cruise_label(cruise))
            else:
                ax1.scatter(y_train[mask], y_pred_train[mask], alpha=0.6, 
                           color=cruise_colors[cruise], s=20, 
                           label=get_cruise_label(cruise))
        
        ax1.legend(title='Training Data', loc='upper left', fontsize=8, title_fontsize=9)
        # add one-to-one line to legend
        ax1.plot([], [], 'k--', lw=2, label='1:1 Line')
        ax1.legend(title='Training Data', loc='upper left', fontsize=8, title_fontsize=9)
    else:
        if train_uncertainty_orig is not None:
            ax1.errorbar(y_train, y_pred_train,
                        xerr=y_train * INSITU_ERROR_FRAC,
                        yerr=train_uncertainty_orig,
                        fmt='o', alpha=0.6, color='blue', markersize=4,
                        capsize=None, capthick=0.5, elinewidth=0.5)
        else:
            ax1.scatter(y_train, y_pred_train, alpha=0.6, color='blue', s=20)
    
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    ax1.set_xlabel('Ship-based measured DiatC (mg m⁻³)', fontsize=14)
    ax1.set_ylabel('Random forest predicted DiatC (mg m⁻³)', fontsize=14)
    ax1.set_title(f'Training Data', fontsize=18)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks([0.001, 0.01, 0.1, 1, 10, 100])
    ax1.set_yticks([0.001, 0.01, 0.1, 1, 10, 100])
    ax1.set_xticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])
    ax1.set_yticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])
    # set figure fontsize
    plt.setp(ax1.get_xticklabels(), fontsize=14)
    plt.setp(ax1.get_yticklabels(), fontsize=14)

    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.0002, 400)
    ax1.set_ylim(0.0002, 400)
    
    # Test plot with error bars
    if test_cruises is not None:
        unique_test_cruises = sorted(test_cruises.unique())
        
        # Plot each cruise with its specific color and error bars
        for cruise in unique_test_cruises:
            mask = test_cruises == cruise
            if test_uncertainty_orig is not None:
                ax2.errorbar(y_test[mask], y_pred_test[mask], 
                           xerr=y_test[mask] * INSITU_ERROR_FRAC,
                           yerr=test_uncertainty_orig[mask],
                           fmt='o', alpha=0.6,
                           color=cruise_colors[cruise], ecolor=cruise_colors[cruise],
                           markerfacecolor='none', markeredgecolor=cruise_colors[cruise],
                           markersize=6, capsize=None, capthick=0.5,
                           elinewidth=0.5, label=get_cruise_label(cruise))
            else:
                ax2.scatter(y_test[mask], y_pred_test[mask], alpha=0.6,
                           facecolors='none', edgecolors=cruise_colors[cruise], s=40, marker='o',
                           label=get_cruise_label(cruise))
        
        ax2.legend(title='Validation Data', loc='upper left', fontsize=8, title_fontsize=9)
        # add one-to-one line to legend
        ax2.plot([], [], 'k--', lw=2, label='1:1 Line')
        ax2.legend(title='Validation Data', loc='upper left', fontsize=8, title_fontsize=9)
    else:
        if test_uncertainty_orig is not None:
            ax2.errorbar(y_test, y_pred_test,
                        xerr=y_test * INSITU_ERROR_FRAC,
                        yerr=test_uncertainty_orig,
                        fmt='o', alpha=0.6,
                        color='red', ecolor='red',
                        markerfacecolor='none', markeredgecolor='red', markersize=6,
                        capsize=None, capthick=0.5, elinewidth=0.5)
        else:
            ax2.scatter(y_test, y_pred_test, alpha=0.6, facecolors='none', edgecolors='red', s=40, marker='o')
    
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax2.set_xlabel('Ship-based measured DiatC (mg m⁻³)', fontsize=14)
    ax2.set_ylabel('Random forest predicted DiatC (mg m⁻³)', fontsize=14)
    ax2.set_title(f'Validation Data', fontsize=18)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xticks([0.001, 0.01, 0.1, 1, 10, 100])
    ax2.set_yticks([0.001, 0.01, 0.1, 1, 10, 100])
    ax2.set_xticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])
    ax2.set_yticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])
    # set figure fontsize
    plt.setp(ax2.get_xticklabels(), fontsize=14)
    plt.setp(ax2.get_yticklabels(), fontsize=14)    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.0002, 100)
    ax2.set_ylim(0.0002, 100)

    plt.tight_layout()
    
    # Save prediction plots with error bars
    figures_dir = Path(FIGURES_DIR)
    figures_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(figures_dir / 'predictions_comparison_with_uncertainty.png', dpi=300, bbox_inches='tight')
    print(f"Prediction plots with uncertainty saved to {figures_dir / 'predictions_comparison_with_uncertainty.png'}")
    plt.close()  # Close to avoid interactive display

def plot_uncertainty_distribution(uncertainty_results, y_pred_train, y_pred_test, 
                                 train_cruises=None, test_cruises=None):
    """
    Create plots showing the distribution and characteristics of prediction uncertainties.
    """
    if not uncertainty_results['has_uncertainty']:
        return
        
    train_uncertainty = uncertainty_results['train_uncertainty']
    test_uncertainty = uncertainty_results['test_uncertainty']
    
    if train_uncertainty is None or test_uncertainty is None:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Convert uncertainties to original scale if needed
    if TARGET_LOG10:
        conversion_factor = np.log(10)
        train_uncertainty_orig = y_pred_train * conversion_factor * train_uncertainty 
        test_uncertainty_orig = y_pred_test * conversion_factor * test_uncertainty
    else:
        train_uncertainty_orig = train_uncertainty
        test_uncertainty_orig = test_uncertainty
    
    # Plot 1: Histogram of uncertainties
    ax1.hist(train_uncertainty_orig, bins=30, alpha=0.7, label='Training', density=True)
    ax1.hist(test_uncertainty_orig, bins=30, alpha=0.7, label='Validation', density=True)
    ax1.set_xlabel('Prediction Uncertainty (mg m⁻³)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Prediction Uncertainties')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Uncertainty vs Prediction magnitude
    ax2.scatter(y_pred_train, train_uncertainty_orig, alpha=0.5, s=15, label='Training')
    ax2.scatter(y_pred_test, test_uncertainty_orig, alpha=0.5, s=15, label='Validation')
    ax2.set_xlabel('Predicted DiatC (mg m⁻³)')
    ax2.set_ylabel('Prediction Uncertainty (mg m⁻³)')
    ax2.set_title('Uncertainty vs Prediction Magnitude')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Coefficient of variation (uncertainty/prediction)
    train_cv = train_uncertainty_orig / y_pred_train
    test_cv = test_uncertainty_orig / y_pred_test
    
    ax3.hist(train_cv, bins=30, alpha=0.7, label='Training', density=True)
    ax3.hist(test_cv, bins=30, alpha=0.7, label='Validation', density=True)
    ax3.set_xlabel('Coefficient of Variation (Uncertainty/Prediction)')
    ax3.set_ylabel('Density')
    ax3.set_title('Relative Uncertainty Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Uncertainty by cruise (if available)
    if train_cruises is not None and test_cruises is not None:
        cruise_colors = get_cruise_color_mapping(train_cruises, test_cruises)
        
        # Box plots of uncertainty by cruise
        all_cruises = []
        all_uncertainties = []
        all_colors = []
        
        for cruise in sorted(train_cruises.unique()):
            mask = train_cruises == cruise
            all_cruises.extend([f'Train-{int(cruise)}'] * np.sum(mask))
            all_uncertainties.extend(train_uncertainty_orig[mask])
            all_colors.extend([cruise_colors[cruise]] * np.sum(mask))
            
        for cruise in sorted(test_cruises.unique()):
            mask = test_cruises == cruise
            all_cruises.extend([f'Test-{int(cruise)}'] * np.sum(mask))
            all_uncertainties.extend(test_uncertainty_orig[mask])
            all_colors.extend([cruise_colors[cruise]] * np.sum(mask))
        
        # Create box plot
        unique_cruises = sorted(set(all_cruises), key=lambda x: (x.split('-')[0], int(x.split('-')[1])))
        cruise_data = []
        cruise_labels = []
        
        for cruise_label in unique_cruises:
            mask = [c == cruise_label for c in all_cruises]
            cruise_data.append([all_uncertainties[i] for i, m in enumerate(mask) if m])
            cruise_labels.append(cruise_label)
        
        bp = ax4.boxplot(cruise_data, labels=cruise_labels)
        ax4.set_xlabel('Cruise')
        ax4.set_ylabel('Prediction Uncertainty (mg m⁻³)')
        ax4.set_title('Uncertainty by Cruise')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
    else:
        # Summary statistics table
        stats_text = (f"Uncertainty Statistics:\n\n"
                      f"Training Set:\n"
                      f"  Mean: {train_uncertainty_orig.mean():.4f} mg m⁻³\n"
                      f"  Median: {np.median(train_uncertainty_orig):.4f} mg m⁻³\n"
                      f"  Std: {train_uncertainty_orig.std():.4f} mg m⁻³\n\n"
                      f"Test Set:\n"
                      f"  Mean: {test_uncertainty_orig.mean():.4f} mg m⁻³\n"
                      f"  Median: {np.median(test_uncertainty_orig):.4f} mg m⁻³\n"
                      f"  Std: {test_uncertainty_orig.std():.4f} mg m⁻³\n\n"
                      f"Relative Uncertainty:\n"
                      f"  Mean CV (train): {train_cv.mean():.4f}\n"
                      f"  Mean CV (test): {test_cv.mean():.4f}")
        
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='center', fontfamily='monospace')
        ax4.set_title('Uncertainty Statistics')
        ax4.axis('off')
    
    plt.tight_layout()
    
    # Save uncertainty analysis plots
    figures_dir = Path(FIGURES_DIR)
    figures_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(figures_dir / 'uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Uncertainty analysis plots saved to {figures_dir / 'uncertainty_analysis.png'}")
    plt.close()  # Close to avoid interactive display

def plot_baseline_comparison(y_test_actual, baseline_diatC, test_cruises=None, cruise_colors=None, chla=None):
    """
    Create scatter plot comparing actual DiatC values vs Chase et al. 2022 baseline predictions.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # 27% measurement error on in situ DiatC values (x-axis), based on Chase et al. 2022 (https://doi.org/10.1029/2022GL098076)
    INSITU_ERROR_FRAC = 0.27
    
    # Calculate baseline error range if chla values are provided
    baseline_yerr_lower = None
    baseline_yerr_upper = None
    if chla is not None:
        baseline_min = 0.9 * chla**1.6
        baseline_max = 2.1 * chla**2.2
        # Ensure error bars are positive distances from the center point
        baseline_yerr_lower = np.maximum(baseline_diatC - baseline_min, 0.0)  # Distance to lower bound (ensure positive)
        baseline_yerr_upper = np.maximum(baseline_max - baseline_diatC, 0.0)  # Distance to upper bound (ensure positive)
    
    # Plot actual values vs baseline predictions
    if test_cruises is not None and cruise_colors is not None:
        unique_test_cruises = sorted(test_cruises.unique())
        
        # Plot each cruise with its specific color and error bars
        for cruise in unique_test_cruises:
            mask = test_cruises == cruise
            # Create asymmetric y-error bars if baseline error range is available
            yerr_asym = None
            if baseline_yerr_lower is not None and baseline_yerr_upper is not None:
                yerr_asym = [baseline_yerr_lower[mask], baseline_yerr_upper[mask]]
            
            ax.errorbar(y_test_actual[mask], baseline_diatC[mask], 
                       xerr=y_test_actual[mask] * INSITU_ERROR_FRAC,
                       yerr=yerr_asym,
                       fmt='o', alpha=0.6,
                       color=cruise_colors[cruise], ecolor=cruise_colors[cruise],
                       markerfacecolor='none', markeredgecolor=cruise_colors[cruise],
                       markersize=6, capsize=None, capthick=0.5,
                       elinewidth=0.5, label=get_cruise_label(cruise))
        
        ax.legend(title='Validation Data', loc='upper left', fontsize=8, title_fontsize=9)
        ax.plot([], [], 'k--', lw=2, label='1:1 Line')
        ax.legend(title='Validation Data', loc='upper left', fontsize=8, title_fontsize=9)
    else:
        # Create asymmetric y-error bars if baseline error range is available
        yerr_asym = None
        if baseline_yerr_lower is not None and baseline_yerr_upper is not None:
            yerr_asym = [baseline_yerr_lower, baseline_yerr_upper]
        
        ax.errorbar(y_test_actual, baseline_diatC,
                   xerr=y_test_actual * INSITU_ERROR_FRAC,
                   yerr=yerr_asym,
                   fmt='o', alpha=0.6,
                   color='blue', ecolor='blue',
                   markerfacecolor='none', markeredgecolor='blue', markersize=6,
                   capsize=None, capthick=0.5, elinewidth=0.5)
    
    # 1:1 line
    min_val = min(baseline_diatC.min(), y_test_actual.min())
    max_val = max(baseline_diatC.max(), y_test_actual.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    ax.set_xlabel('Ship-based measured DiatC (mg m⁻³)', fontsize=14)
    ax.set_ylabel('Baseline predicted DiatC (mg m⁻³)', fontsize=14)
    ax.set_title('Validation Data - Baseline', fontsize=18)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([0.001, 0.01, 0.1, 1, 10, 100])
    ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 100])
    ax.set_xticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])
    ax.set_yticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0002, 100)
    ax.set_ylim(0.0002, 100)
    
    plt.tight_layout()
    
    # Save baseline comparison plot
    figures_dir = Path(FIGURES_DIR)
    figures_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(figures_dir / 'baseline_comparison_with_uncertainty.png', dpi=300, bbox_inches='tight')
    print(f"Baseline comparison plot saved to {figures_dir / 'baseline_comparison_with_uncertainty.png'}")
    plt.close()  # Close to avoid interactive display

def plot_density_comparison(y_pred, y_actual, baseline_values):
    """
    Create density plot comparing predicted, actual, and baseline values.
    """
    import seaborn as sns
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(np.log10(y_pred), label='RF Predicted Values', fill=True)
    sns.kdeplot(np.log10(y_actual), label='Actual Values', fill=True)
    sns.kdeplot(np.log10(baseline_values), label='Baseline Values', fill=True)
    plt.title('Density Plot of Predicted vs. Actual vs. Baseline Values (Log Scale)')
    plt.xlabel('Log10 Values')
    plt.ylabel('Density')
    plt.legend()
    
    # Save density plot
    figures_dir = Path(FIGURES_DIR)
    figures_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(figures_dir / 'density_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Density comparison plot saved to {figures_dir / 'density_comparison.png'}")
    plt.close()  # Close to avoid interactive display

def plot_permutation_importance(permutation_importances, feature_names):
    """
    Create a box plot showing permutation importance distributions across CV folds.
    """
    # Convert list of importance arrays to dataframe for easier plotting
    importance_data = []
    for fold_idx, fold_importances in enumerate(permutation_importances):
        for feature_idx, feature_name in enumerate(feature_names):
            for importance_value in fold_importances[feature_idx]:
                importance_data.append({
                    'Feature': feature_name,
                    'Importance': importance_value,
                    'Fold': fold_idx + 1
                })
    
    importance_df = pd.DataFrame(importance_data)
    
    # Calculate feature importance order (highest to lowest)
    feature_means = importance_df.groupby('Feature')['Importance'].mean()
    feature_order = feature_means.sort_values(ascending=False).index.tolist()
    
    # Create box plot
    plt.figure(figsize=(12, 8))
    box_plot = sns.boxplot(data=importance_df, x='Feature', y='Importance', 
                          order=feature_order)
    
    # Make boxes transparent by modifying the patch objects
    for patch in box_plot.patches:
        patch.set_alpha(0.6)
    
    plt.title('Permutation Importance Distribution Across CV Folds (Ordered by Importance)')
    plt.xlabel('Features (Ordered by Mean Importance)')
    plt.ylabel('Permutation Importance (MAE decrease)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    figures_dir = Path(FIGURES_DIR)
    figures_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(figures_dir / 'permutation_importance.png', dpi=300, bbox_inches='tight')
    print(f"Permutation importance plot saved to {figures_dir / 'permutation_importance.png'}")
    plt.close()  # Close to avoid interactive display
    
    # Also save the raw data to CSV
    summary_stats = importance_df.groupby('Feature')['Importance'].agg(['mean', 'std', 'median'])
    summary_stats = summary_stats.sort_values('mean', ascending=False)
    importance_file = Path(FIGURES_DIR) / 'permutation_importance_summary.csv'
    summary_stats.to_csv(importance_file)
    print(f"Permutation importance summary saved to {importance_file}")
    
    return importance_df

def plot_error_distribution(y_test, y_pred_test):
    """
    Plot the distribution of errors on the test dataset.
    """
    errors = y_pred_test - y_test
    
    # Calculate the absolute errors
    abs_errors = np.abs(errors)
    
    # Calculate the percentage of errors within 1 of zero
    within_1 = np.sum(abs_errors <= 1) / len(errors) * 100
    
    # Calculate the percentage of errors within 3 of zero
    within_3 = np.sum(abs_errors <= 3) / len(errors) * 100
    
    # Output the results
    print(f"\\nError Distribution Analysis:")
    print(f"Percentage of errors within 1 of zero: {within_1:.2f}%")
    print(f"Percentage of errors within 3 of zero: {within_3:.2f}%")
    
    # Plot the distribution of errors
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, bins=50, kde=False, color='blue')
    
    # Add titles and labels
    plt.title('Distribution of Errors on Test Dataset', fontsize=16)
    plt.xlabel('Error (Predicted DiatC - Measured DiatC)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xlim(-40, 20)
    
    # Save error distribution plot
    figures_dir = Path(FIGURES_DIR)
    figures_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(figures_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Error distribution plot saved to {figures_dir / 'error_distribution.png'}")
    plt.close()  # Close to avoid interactive display

def plot_learning_curves(estimator, X, y, groups, model_name="Model"):
    """Plot learning curves to diagnose overfitting"""
    print(f"\nGenerating learning curves for {model_name}...")
    
    # Ensure groups is a numpy array and handle potential issues
    groups = np.asarray(groups).ravel()
    
    # Check if we have enough groups for CV
    n_unique_groups = len(np.unique(groups))
    n_splits = min(3, n_unique_groups - 1) if n_unique_groups > 1 else 1
    
    if n_splits < 2:
        print(f"WARNING: Only {n_unique_groups} unique groups, skipping learning curves")
        return
    
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, groups=groups, 
            cv=GroupKFold(n_splits=n_splits),
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1
        )
        
        # Convert to positive MSE and then to RMSE for better interpretability
        train_rmse = np.sqrt(-train_scores)
        val_rmse = np.sqrt(-val_scores)
        
        # Calculate means and standard deviations
        train_rmse_mean = np.mean(train_rmse, axis=1)
        train_rmse_std = np.std(train_rmse, axis=1)
        val_rmse_mean = np.mean(val_rmse, axis=1)
        val_rmse_std = np.std(val_rmse, axis=1)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot training scores
        plt.plot(train_sizes, train_rmse_mean, 'o-', color='blue', 
                label=f'Training RMSE', linewidth=2, markersize=6)
        plt.fill_between(train_sizes, train_rmse_mean - train_rmse_std, 
                        train_rmse_mean + train_rmse_std, alpha=0.2, color='blue')
        
        # Plot test scores
        plt.plot(train_sizes, val_rmse_mean, 'o-', color='red', 
                label=f'Test RMSE', linewidth=2, markersize=6)
        plt.fill_between(train_sizes, val_rmse_mean - val_rmse_std, 
                        val_rmse_mean + val_rmse_std, alpha=0.2, color='red')
        
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('RMSE (Log Scale)', fontsize=12)
        plt.title(f'Learning Curves - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add text annotations for overfitting analysis
        final_train_rmse = train_rmse_mean[-1]
        final_val_rmse = val_rmse_mean[-1]
        gap = final_val_rmse - final_train_rmse
        
        plt.text(0.02, 0.98, 
                f'Final Training RMSE: {final_train_rmse:.4f}\n'
                f'Final Test RMSE: {final_val_rmse:.4f}\n'
                f'Overfitting Gap: {gap:.4f}',
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save learning curves plot
        figures_dir = Path(FIGURES_DIR)
        figures_dir.mkdir(exist_ok=True, parents=True)
        
        # Clean model name for filename
        clean_model_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        filename = f'learning_curves_{clean_model_name}.png'
        plt.savefig(figures_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to {figures_dir / filename}")
        
        plt.close()  # Close to avoid interactive display
        
        # Return metrics for analysis
        return {
            'final_train_rmse': final_train_rmse,
            'final_test_rmse': final_val_rmse, 
            'overfitting_gap': gap,
            'train_sizes': train_sizes,
            'train_rmse_mean': train_rmse_mean,
            'val_rmse_mean': val_rmse_mean
        }
        
    except Exception as e:
        print(f"Error generating learning curves: {e}")
        return None

def plot_cruise_map(data, train_cruises, test_cruises, cruise_colors=None):
    """
    Create a world map plot showing the geographical distribution of cruise locations,
    with training and test cruises distinguished by different markers.
    Uses the original full dataset to access lat/lon information.
    """
    print(f"\nCreating cruise location map...")
    
    # Check if lat/lon columns exist
    if 'lat' not in data.columns or 'lon' not in data.columns:
        print("WARNING: lat/lon columns not found, skipping map plot")
        return
    
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        use_cartopy = True
    except ImportError:
        print("WARNING: cartopy not installed, falling back to basic map")
        use_cartopy = False
    
    if use_cartopy:
        # Create figure with map projection
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

        # Add land, ocean, and coastlines
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
        ax.add_feature(cfeature.OCEAN, facecolor='white', alpha=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
        #ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
        #ax.add_feature(cfeature.LAKES, alpha=0.5)
        #ax.add_feature(cfeature.RIVERS, linewidth=0.3)

        # Set global extent
        ax.set_global()
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

    else:
        # No synthetic fallback: when cartopy is unavailable, render a plain map canvas
        # without fake land polygons so users know the plotting dependency is missing.
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_facecolor('lightblue')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Longitude (°E)', fontsize=12)
        ax.set_ylabel('Latitude (°N)', fontsize=12)
        print("WARNING: cartopy is not installed; continents and coastlines will not be rendered.")
    
    # Get consistent cruise color mapping (same as other plots)
    if cruise_colors is None:
        cruise_colors = get_cruise_color_mapping(train_cruises, test_cruises)
    
    # Plot training cruises
    train_data = data[data['cruise'].isin(train_cruises)].dropna(subset=['lat', 'lon'])
    for cruise in train_cruises:
        cruise_data = train_data[train_data['cruise'] == cruise]
        if len(cruise_data) > 0:
            ax.scatter(cruise_data['lon'], cruise_data['lat'], 
                      c=[cruise_colors[cruise]], s=30, alpha=0.8, 
                      marker='o', edgecolors=None, linewidth=0.2,
                      label=get_cruise_label(cruise),
                      transform=ccrs.PlateCarree() if use_cartopy else None,
                      zorder=5)
    
    # Plot test cruises
    test_data = data[data['cruise'].isin(test_cruises)].dropna(subset=['lat', 'lon'])
    for cruise in test_cruises:
        cruise_data = test_data[test_data['cruise'] == cruise]
        if len(cruise_data) > 0:
            ax.scatter(cruise_data['lon'], cruise_data['lat'], 
                      c=cruise_colors[cruise], s=80, alpha=0.9, 
                      marker='+', linewidths=0.2,
                      label=get_cruise_label(cruise),
                      transform=ccrs.PlateCarree() if use_cartopy else None,
                      zorder=6)
    
    # Formatting
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    #ax.set_title('Cruise Locations for Diatom Carbon Model Training/Testing', 
    #            fontsize=14, fontweight='bold')
    
    # Create legend with training vs test distinction
    handles, labels = ax.get_legend_handles_labels()
    
    # Separate training and test entries based on cruise membership
    train_handles = []
    test_handles = []
    train_labels = []
    test_labels = []
    
    for handle, label in zip(handles, labels):
        # Attempt to extract cruise number from label using multiple strategies
        import re
        cruise_num = None
        # 1) 'Dataset 23'
        match = re.search(r'Dataset (\d+)', label)
        if match:
            cruise_num = int(match.group(1))
        else:
            # 2) number in parentheses e.g. 'Name (23)'
            match = re.search(r'\((\d+)\)', label)
            if match:
                cruise_num = int(match.group(1))
            else:
                # 3) reverse lookup in CRUISE_NAME_MAP if available
                if CRUISE_NAME_MAP:
                    for k, v in CRUISE_NAME_MAP.items():
                        if str(v) == label:
                            try:
                                cruise_num = int(k)
                                break
                            except Exception:
                                cruise_num = None
                # 4) fallback: any digits in label
                if cruise_num is None:
                    match = re.search(r'(\d+)', label)
                    if match:
                        cruise_num = int(match.group(1))

        if cruise_num is not None:
            if cruise_num in train_cruises:
                train_handles.append(handle)
                train_labels.append(label)
            elif cruise_num in test_cruises:
                test_handles.append(handle)
                test_labels.append(label)
    
    # Sort training entries by cruise number
    def extract_cruise_number(label):
        # Robust extraction of cruise number from various label formats
        import re
        # 1) 'Dataset 23'
        match = re.search(r'Dataset (\d+)', label)
        if match:
            return int(match.group(1))
        # 2) number in parentheses
        match = re.search(r'\((\d+)\)', label)
        if match:
            return int(match.group(1))
        # 3) reverse lookup in CRUISE_NAME_MAP
        if CRUISE_NAME_MAP:
            for k, v in CRUISE_NAME_MAP.items():
                if str(v) == label:
                    try:
                        return int(k)
                    except Exception:
                        continue
        # 4) any digits in label
        match = re.search(r'(\d+)', label)
        return int(match.group(1)) if match else 0
    
    # Sort training entries numerically by cruise number
    if train_handles:
        train_sorted = sorted(zip(train_handles, train_labels), key=lambda x: extract_cruise_number(x[1]))
        train_handles_sorted, train_labels_sorted = zip(*train_sorted)
    else:
        train_handles_sorted, train_labels_sorted = [], []
    
    # Sort testing entries numerically by cruise number  
    if test_handles:
        test_sorted = sorted(zip(test_handles, test_labels), key=lambda x: extract_cruise_number(x[1]))
        test_handles_sorted, test_labels_sorted = zip(*test_sorted)
    else:
        test_handles_sorted, test_labels_sorted = [], []
    
    # Create legend with sections
    def get_color_for_label(label):
        cruise_num = extract_cruise_number(label)
        return cruise_colors.get(cruise_num, cruise_colors.get(str(cruise_num), 'black'))

    train_proxy_handles = [
        Line2D([], [], marker='o', linestyle='None',
               markerfacecolor=get_color_for_label(label),
               markeredgecolor=get_color_for_label(label),
               markersize=10, markeredgewidth=1.8)
        for label in train_labels_sorted
    ]
    test_proxy_handles = [
        Line2D([], [], marker='+', linestyle='None',
               markerfacecolor='none',
               markeredgecolor=get_color_for_label(label),
               markersize=10, markeredgewidth=1.8)
        for label in test_labels_sorted
    ]

    if train_handles_sorted:
        legend1 = ax.legend(train_proxy_handles, train_labels_sorted, 
                           title='Training Data', loc='upper right', 
                           fontsize=12, title_fontsize=14,
                           markerscale=1, framealpha=0.9)
        ax.add_artist(legend1)
    
    if test_handles_sorted:
        legend2 = ax.legend(test_proxy_handles, test_labels_sorted, 
                           title='Validation Data', loc='lower right', 
                           fontsize=12, title_fontsize=14,
                           markerscale=1, framealpha=0.9)
        ax.add_artist(legend2)
    summary_text = (f'Training Samples: {len(train_data)}\n'
                   f'Validation Samples: {len(test_data)}')
    
    # ax.text(0.98, 0.02, summary_text, transform=ax.transAxes, 
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
    #         verticalalignment='bottom', horizontalalignment='right', fontsize=12)
    
    # Save plot
    figures_dir = Path(FIGURES_DIR)
    figures_dir.mkdir(exist_ok=True, parents=True)

    plt.tight_layout()
    plt.savefig(figures_dir / 'cruise_locations_map.png', dpi=300, bbox_inches='tight')
    print(f"Cruise location map saved to {figures_dir / 'cruise_locations_map.png'}")
    plt.close()

def cross_validation_analysis(X_train, y_train, train_data, train_prep_result):
    """
    Perform cross-validation analysis on the training data using GroupKFold by cruise.
    """
    print("\\nPerforming cross-validation analysis...")
    
    # Since prepare_training_xy was called with drop_invalid=True,
    # we need to call it again with drop_invalid=False to get the true valid_mask
    cfg = PrepConfig(
        feature_names=FEATURE_NAMES,
        var_to_log=VAR_TO_LOG,
        target_name=TARGET_NAME,
        target_log10=TARGET_LOG10,
        ratio_defs={
            'chlb_a': ('chlb', 'chla'),
            'chlc_a': ('chlc', 'chla'),
            'ppc_a': ('ppc', 'chla')
        },
        require_positive_for_log=True
    )
    
    # Get the full prepared data without dropping invalid rows
    X_full, y_full, prep_result = prepare_training_xy(train_data, cfg, drop_invalid=False)
    valid_mask = prep_result.valid_mask
    
    print(f"Total rows in train_data: {len(train_data)}")
    print(f"Valid rows after preparation: {valid_mask.sum()}")
    print(f"X_train shape: {X_train.shape}")
    
    # Verify the dimensions match
    if valid_mask.sum() != len(X_train):
        print(f"WARNING: Valid rows ({valid_mask.sum()}) != X_train rows ({len(X_train)})")
        print("Skipping cross-validation due to data inconsistency")
        return np.array([])
    
    # Get the aligned training data (only valid rows)
    train_data_valid = train_data.loc[valid_mask].reset_index(drop=True)
    groups_train = train_data_valid['cruise']
    
    print(f"Unique cruises in valid training data: {sorted(groups_train.unique())}")
    
    # Initialize GroupKFold to split the data in a \"group-aware\" way
    # to avoid data leakage and overtraining
    gkf = GroupKFold(n_splits=5)
    
    # Check if we have enough groups for the splits
    n_groups = len(groups_train.unique())
    if n_groups < 5:
        print(f"WARNING: Only {n_groups} unique cruises available, using {n_groups-1} splits")
        gkf = GroupKFold(n_splits=n_groups-1 if n_groups > 1 else 1)
    
    # Return the number of splitting iterations
    n_splits_actual = gkf.get_n_splits(X_train, y_train, groups_train)
    print(f"Number of CV splits: {n_splits_actual}")
    
    if n_splits_actual == 0:
        print("No splits possible, skipping cross-validation")
        return np.array([])
    
    # Make groups a 1D numpy array
    groups = np.asarray(groups_train).ravel()
    
    # Initialize metric lists for log and original space
    mse_log_list, rmse_log_list = [], []
    mse_orig_list, rmse_orig_list = [], []
    mae_orig_list, medae_orig_list = [], []
    mape_list, mdape_list = [], []
    
    # Additional info for CSV export
    fold_info = []
    
    # Initialize permutation importance tracking
    permutation_importances = []  # List of importance arrays for each fold
    
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_train, y_train, groups=groups), start=1):
        # indices here are positional for X_train/y_train -> use .iloc for pandas
        X_tr = X_train.iloc[tr_idx]
        X_te = X_train.iloc[te_idx]

        # y may be Series or numpy
        if hasattr(y_train, "iloc"):
            y_tr = y_train.iloc[tr_idx]
            y_te = y_train.iloc[te_idx]
        else:
            y_tr = y_train[tr_idx]
            y_te = y_train[te_idx]

        # Create new model instance for each fold
        model = create_model()
        
        model.fit(X_tr, y_tr)
            
        y_pred = model.predict(X_te)
        
        # Calculate permutation importance for this fold
        perm_importance = permutation_importance(
            model, X_te, y_te, 
            n_repeats=10,  # Number of times to permute each feature
            random_state=42 + fold,  # Different seed for each fold
            scoring='neg_mean_absolute_error'  # Use MAE as scoring metric
        )
        permutation_importances.append(perm_importance.importances)

        # --- log10-space metrics ---
        mse_log = mean_squared_error(y_te, y_pred)
        rmse_log = np.sqrt(mse_log)

        # --- original-space metrics  ---
        y_te_orig = np.power(10, np.asarray(y_te))
        y_pred_orig = np.power(10, np.asarray(y_pred))
        mae_orig = mean_absolute_error(y_te_orig, y_pred_orig)
        medae_orig = median_absolute_error(y_te_orig, y_pred_orig)
        
        mse_orig = mean_squared_error(y_te_orig, y_pred_orig)
        rmse_orig = np.sqrt(mse_orig)

        mape = np.mean(np.abs((y_te_orig - y_pred_orig) / y_te_orig)) * 100
        mdape = np.median(np.abs((y_te_orig - y_pred_orig) / y_te_orig)) * 100

        mse_log_list.append(mse_log)
        rmse_log_list.append(rmse_log)
        mse_orig_list.append(mse_orig)
        rmse_orig_list.append(rmse_orig)
        mae_orig_list.append(mae_orig)
        medae_orig_list.append(medae_orig)
        mape_list.append(mape)
        mdape_list.append(mdape)
        held_out = np.unique(groups[te_idx])
        
        # Store fold information for CSV export
        fold_info.append({
            'fold': fold,
            'n_test': len(te_idx),
            'n_held_out_cruises': len(held_out),
            'held_out_cruises': str(held_out),
            'mse_log': mse_log,
            'rmse_log': rmse_log,
            'mse_orig': mse_orig,
            'rmse_orig': rmse_orig,
            'mae_orig': mae_orig,
            'medae_orig': medae_orig,
            'mape': mape,
            'mdape': mdape
        })
        
        print(
            f"Fold {fold}: test n={len(te_idx):4d}, held-out cruises={len(held_out):2d}, "
            f"MSE_log={mse_log:.6g}, RMSE_log={rmse_log:.6g}, "
            f"RMSE_orig={rmse_orig:.6g}"
        )

    print("\\n=== Cruise-holdout CV summary (across folds) ===")
    print(f"MSE_log   : mean={np.mean(mse_log_list):.6g}  std={np.std(mse_log_list):.6g}")
    print(f"RMSE_log  : mean={np.mean(rmse_log_list):.6g} std={np.std(rmse_log_list):.6g}")
    print(f"MSE_orig  : mean={np.mean(mse_orig_list):.6g} std={np.std(mse_orig_list):.6g}")
    print(f"RMSE_orig : mean={np.mean(rmse_orig_list):.6g} std={np.std(rmse_orig_list):.6g}")
    print(f"MAE_orig  : mean={np.mean(mae_orig_list):.6g} std={np.std(mae_orig_list):.6g}")
    print(f"MedAE_orig: mean={np.mean(medae_orig_list):.6g} std={np.std(medae_orig_list):.6g}")
    print(f"MAPE      : mean={np.mean(mape_list):.6g} std={np.std(mape_list):.6g}")
    print(f"MdAPE     : mean={np.mean(mdape_list):.6g} std={np.std(mdape_list):.6g}")
    
    # Save detailed fold metrics to CSV
    fold_df = pd.DataFrame(fold_info)
    cv_metrics_file = '../figures/cross_validation_metrics.csv'
    fold_df.to_csv(cv_metrics_file, index=False)
    print(f"\\nCross-validation fold metrics saved to {cv_metrics_file}")
    
    # Also save summary statistics 
    summary_stats = {
        'metric': ['MSE_log', 'RMSE_log', 'MSE_orig', 'RMSE_orig', 'MAE_orig', 'MedAE_orig', 'MAPE', 'MdAPE'],
        'mean': [np.mean(mse_log_list), np.mean(rmse_log_list), np.mean(mse_orig_list), 
                 np.mean(rmse_orig_list), np.mean(mae_orig_list), np.mean(medae_orig_list), np.mean(mape_list), np.mean(mdape_list)],
        'std': [np.std(mse_log_list), np.std(rmse_log_list), np.std(mse_orig_list),
                np.std(rmse_orig_list), np.std(mae_orig_list), np.std(medae_orig_list), np.std(mape_list), np.std(mdape_list)]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_file = '../figures/cross_validation_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Cross-validation summary statistics saved to {summary_file}")
    
    # Return MAE original scores and permutation importances for compatibility
    cv_scores = np.array(mae_orig_list)
    
    return cv_scores, permutation_importances

def main(argv=None):
    """
    Main evaluation workflow.
    """
    args = parse_args(argv)

    global DATA_PATH, FIGURES_DIR
    DATA_PATH = str(Path(args.training_data).expanduser().resolve())
    FIGURES_DIR = str(Path(args.figures_dir).expanduser().resolve())

    print("=== Diatom Carbon Model Evaluation ===")
    print(f"Using configuration from model_config:")
    print(f"  Model Type: {MODEL_TYPE.upper()}")
    print(f"  Features: {FEATURE_NAMES}")
    print(f"  Target: {TARGET_NAME}")
    print(f"  Log-transform variables: {VAR_TO_LOG}")
    print(f"  Log-transform target: {TARGET_LOG10}")
    print()
    
    # Load and prepare data
    data, cfg = load_and_prepare_data(DATA_PATH)
    
    # Select required columns - need base columns for ratio computation
    required_cols = ['cruise'] + FEATURE_NAMES + [TARGET_NAME]
    
    # Add base columns needed for ratio computation
    if RATIO_DEFS:
        for ratio_name, (num_col, den_col) in RATIO_DEFS.items():
            if num_col not in required_cols:
                required_cols.append(num_col)
            if den_col not in required_cols:
                required_cols.append(den_col)
    
    # Add lat/lon columns for map plotting if available
    for geo_col in ['lat', 'lon']:
        if geo_col in data.columns and geo_col not in required_cols:
            required_cols.append(geo_col)
    
    # Add stratum column for stratified sampling (should be available after load_and_prepare_data)
    if 'stratum' in data.columns:
        required_cols.append('stratum')
        print(f"Including 'stratum' column for stratified sampling")
    else:
        print(f"WARNING: 'stratum' column not found - this should not happen!")
    
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
        required_cols = [col for col in required_cols if col in data.columns]
    
    df = data[required_cols].copy()
    
    # Remove rows with NaN cruise values as they can't be properly split
    initial_len = len(df)
    df = df.dropna(subset=['cruise'])
    if len(df) < initial_len:
        print(f"Removed {initial_len - len(df)} rows with missing cruise information")
    
    print(f"Selected {len(df)} rows with required columns")
    print(f"Unique cruises: {sorted(df['cruise'].unique())}")
    
    # Create train/test split
    train_data, test_data, train_cruises, test_cruises = create_train_test_split(df, cfg)
    
    # Create consistent cruise color mapping for all plots
    initial_cruise_colors = get_cruise_color_mapping(train_cruises, test_cruises)
    
    # Plot the distribution of values in different strata for training and test sets
    print("\\nCreating strata distribution plots...")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    # Plot for training data
    train_strata_counts = train_data['stratum'].value_counts().sort_index()
    ax[0].bar(train_strata_counts.index, train_strata_counts.values, color='blue')
    ax[0].set_title('Training Data Strata Distribution')
    ax[0].set_xlabel('Stratum')
    ax[0].set_ylabel('Count')
    ax[0].tick_params(axis='x', rotation=45)
    
    # Plot for test data
    test_strata_counts = test_data['stratum'].value_counts().sort_index()
    ax[1].bar(test_strata_counts.index, test_strata_counts.values, color='green')
    ax[1].set_title('Validation Data Strata Distribution')
    ax[1].set_xlabel('Stratum')
    ax[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save figure (create directory if it doesn't exist)
    figures_dir = Path(FIGURES_DIR)
    figures_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(figures_dir / 'train_test_strata.png', dpi=300, bbox_inches='tight')
    print(f"Strata distribution plot saved to {figures_dir / 'train_test_strata.png'}")
    plt.close()  # Close the figure to avoid interactive display issues
    
    # Plot cruise locations map
    plot_cruise_map(data, train_cruises, test_cruises, initial_cruise_colors)
    
    # Prepare datasets
    X_train, y_train, X_test, y_test, train_prep_result, train_cruises, test_cruises = prepare_datasets(train_data, test_data, cfg)
    
    # Train and evaluate model
    model, predictions, metrics, uncertainty_results = evaluate_model(X_train, y_train, X_test, y_test)
    y_pred_train, y_pred_test = predictions
    train_metrics, test_metrics = metrics
    
    # Save evaluation metrics to CSV
    evaluation_results = {
        'dataset': ['Training', 'Test'],
        'mae': [train_metrics['mae'], test_metrics['mae']],
        'medae': [train_metrics['medae'], test_metrics['medae']],
        'rmse': [train_metrics['rmse'], test_metrics['rmse']],
        'R2': [train_metrics['r2'], test_metrics['r2']],
        'R2_log': [train_metrics['r2_log'], test_metrics['r2_log']],
        'mape': [train_metrics['mape'], test_metrics['mape']],
        'mean_percentage_error': [train_metrics['mean_percentage_error'], test_metrics['mean_percentage_error']],
        'med_percentage_error': [train_metrics['med_percentage_error'], test_metrics['med_percentage_error']],
        'mean_bias_error': [train_metrics['mean_bias_error'], test_metrics['mean_bias_error']],
        'median_bias_error': [train_metrics['median_bias_error'], test_metrics['median_bias_error']],
        'spearman_rho': [train_metrics['spearman_rho'], test_metrics['spearman_rho']],
    }
    evaluation_df = pd.DataFrame(evaluation_results)
    evaluation_file = Path(FIGURES_DIR) / 'evaluation_metrics.csv'
    evaluation_df.to_csv(evaluation_file, index=False)
    print(f"Evaluation metrics saved to {evaluation_file}")
    
    # Calculate baseline using Chase et al. 2022 Eq. 6 (https://doi.org/10.1029/2022GL098076)
    print(f"\\n=== Baseline Comparison (Chase et al. 2022) ===")
    
    # Convert test data back to original scale for baseline comparison
    y_test_orig_baseline = np.power(10, y_test) if TARGET_LOG10 else y_test
    chla = 10**X_test['chla']  # Convert from log space to original chla values
    baseline_diatC = 1.5 * chla**1.9 # Chase et al. 2022 Eq. 6
    #baseline_diatC = 1.04 * chla**2.13 # retuned with all RF model training/testing data
    
    # Calculate baseline metrics using the same calc_metrics function for consistency
    if TARGET_LOG10:
        # For consistent comparison with model evaluation:
        # - y_test is already log10(actual DiatC)
        # - baseline_diatC is in original scale, convert to log for r2_log calculation
        baseline_diatC_log = np.log10(np.maximum(baseline_diatC, 1e-10))  # Avoid log(0)
        y_test_log_baseline = y_test  # Already in log scale
    else:
        baseline_diatC_log = baseline_diatC
        y_test_log_baseline = y_test_orig_baseline
    
    # Debug baseline calculation before calling calc_metrics
    print(f"\\nBaseline Calculation Debug:")
    print(f"  chla range: {chla.min():.4f} to {chla.max():.4f}")
    print(f"  baseline_diatC range: {baseline_diatC.min():.4f} to {baseline_diatC.max():.4f}")
    print(f"  y_test_orig_baseline range: {y_test_orig_baseline.min():.4f} to {y_test_orig_baseline.max():.4f}")
    
    if TARGET_LOG10:
        print(f"  baseline_diatC_log range: {baseline_diatC_log.min():.4f} to {baseline_diatC_log.max():.4f}")
        print(f"  y_test_log_baseline range: {y_test_log_baseline.min():.4f} to {y_test_log_baseline.max():.4f}")
        
        # Check for any potential issues
        if np.any(baseline_diatC <= 0):
            print(f"  WARNING: {np.sum(baseline_diatC <= 0)} baseline_diatC values <= 0")
        if np.any(~np.isfinite(baseline_diatC_log)):
            print(f"  WARNING: {np.sum(~np.isfinite(baseline_diatC_log))} non-finite baseline_diatC_log values")
    
    # Call the exact same calc_metrics function used for model evaluation
    baseline_metrics = calc_metrics(
        y_test_orig_baseline,  # y_true_orig: actual values in original scale
        baseline_diatC,        # y_pred_orig: baseline predictions in original scale  
        y_test_log_baseline,   # y_true_log: actual values in log scale
        baseline_diatC_log,    # y_pred_log: baseline predictions in log scale
        "Baseline"             # dataset_name
    )
    
    # Save baseline metrics to CSV with all metrics matching the test set
    baseline_results = {
        'method': ['Random Forest', 'Chase et al. 2022 Baseline'],
        'mae': [test_metrics['mae'], baseline_metrics['mae']],
        'medae': [test_metrics['medae'], baseline_metrics['medae']],
        'rmse': [test_metrics['rmse'], baseline_metrics['rmse']],
        'R2': [test_metrics['r2'], baseline_metrics['r2']],
        'R2_log': [test_metrics['r2_log'], baseline_metrics['r2_log']],
        'mape': [test_metrics['mape'], baseline_metrics['mape']],
        'mean_percentage_error': [test_metrics['mean_percentage_error'], baseline_metrics['mean_percentage_error']],
        'med_percentage_error': [test_metrics['med_percentage_error'], baseline_metrics['med_percentage_error']],
        'mean_bias_error': [test_metrics['mean_bias_error'], baseline_metrics['mean_bias_error']],
        'median_bias_error': [test_metrics['median_bias_error'], baseline_metrics['median_bias_error']],
        'spearman_rho': [test_metrics['spearman_rho'], baseline_metrics['spearman_rho']]
    }
    baseline_df = pd.DataFrame(baseline_results)
    baseline_file = Path(FIGURES_DIR) / 'model_baseline_comparison.csv'
    baseline_df.to_csv(baseline_file, index=False)
    print(f"Model vs baseline comparison saved to {baseline_file}")
    
    # Plot baseline comparison
    plot_baseline_comparison(
        y_test_orig_baseline,  # Actual DiatC values
        baseline_diatC,        # Baseline predictions
        test_cruises,
        initial_cruise_colors,
        chla  # Chlorophyll-a values for error bar calculation
    )
    
    # Plot density comparison
    plot_density_comparison(
        y_pred_test,           # RF predictions (original scale)
        y_test_orig_baseline,  # Actual values (original scale)
        baseline_diatC         # Baseline predictions
    )
    
    # Plot predictions
    plot_predictions(
        np.power(10, y_train) if TARGET_LOG10 else y_train,
        y_pred_train,
        np.power(10, y_test) if TARGET_LOG10 else y_test, 
        y_pred_test,
        train_metrics, 
        test_metrics,
        train_cruises,
        test_cruises,
        uncertainty_results
    )
    
    # Plot uncertainty analysis if available
    plot_uncertainty_distribution(
        uncertainty_results, 
        y_pred_train, 
        y_pred_test,
        train_cruises, 
        test_cruises
    )
    
    # Plot error distribution on test data
    plot_error_distribution(
        np.power(10, y_test) if TARGET_LOG10 else y_test,
        y_pred_test
    )
    
    # Cross-validation analysis
    cv_scores, permutation_importances = cross_validation_analysis(X_train, y_train, train_data, train_prep_result)
    
    # Generate learning curves to diagnose overfitting
    learning_metrics = plot_learning_curves(
        model, X_train, y_train, 
        train_data.loc[X_train.index, 'cruise'], 
        model_name=f"{MODEL_TYPE.upper()} Model"
    )
    
    # Report learning curve diagnostics
    if learning_metrics:
        print(f"\\n=== Learning Curve Diagnostics ===")
        print(f"Final Training RMSE: {learning_metrics['final_train_rmse']:.4f}")
        print(f"Final Test RMSE: {learning_metrics['final_test_rmse']:.4f}")
        print(f"Overfitting Gap: {learning_metrics['overfitting_gap']:.4f}")
    
    # Plot permutation importance across CV folds
    if permutation_importances:
        plot_permutation_importance(permutation_importances, FEATURE_NAMES)
    
    print("\\n=== Evaluation Complete ===")
    print(f"Model ({MODEL_TYPE.upper()}) uses configuration from model_config.py")
    print(f"Test R²: {test_metrics['r2']:.3f}")
    print(f"Test RMSE: {test_metrics['rmse']:.3f}")
    
    return model, predictions, metrics, cv_scores, learning_metrics

if __name__ == "__main__":
    model, predictions, metrics, cv_scores, learning_metrics = main()