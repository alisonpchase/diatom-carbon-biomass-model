# Diatom Carbon model input and target variable names; defined once here for both CV and final model training to ensure consistency.

# ----------------------------
# Feature definitions
# ----------------------------

FEATURE_NAMES = [
    "chla",      # Chlorophyll a concentration (mg m^-3)
    "chlb",      # Chlorophyll b concentration (mg m^-3)
    "chlc",      # Chlorophyll c1+c2 concentration (mg m^-3)
    "chlb_a",    # ratio of chlorophyll b to chlorophyll a (unitless)
    "chlc_a",    # ratio of chlorophyll c1+c2 to chlorophyll a (unitless)
    "ppc_a",     # ratio of photoprotective carotenoid pigments to chlorophyll a (unitless)
   # "t_avg",     # Temperature average (°C) - High predictive importance (0.2264)
   # "s_avg",     # Salinity average (PSU) - Complementary environmental signal (0.0930)
]

# Derived pigment ratio feature definitions:
# output_feature : (numerator_column, denominator_column)
RATIO_DEFS = {
    "chlb_a": ("chlb", "chla"),
    "chlc_a": ("chlc", "chla"),
    "ppc_a":  ("ppc",  "chla"),  
}

# Variables that are log10-transformed (after ratios are computed)
VAR_TO_LOG = [
    "chla",
    "chlb",
    "chlc",
    "chlb_a",
    "chlc_a",
    "ppc_a",
]

# ----------------------------
# Target definition
# ----------------------------

TARGET_NAME = "diatCarb"
TARGET_LOG10 = True

# ----------------------------
# Model hyperparameters
# ----------------------------

# Model type configuration
MODEL_TYPE = 'rf'  # Random Forest 

RF_PARAMS = dict(
    n_estimators=1500,        # Slightly increased for stable predictions  
    min_samples_split=10,     # Higher to prevent overfitting to small subgroups
    min_samples_leaf=5,       # Increased for robustness across different cruise conditions
    max_features='sqrt',      # Classical choice for better generalization
    max_depth=15,             # Optimized depth based on hyperparameter testing
    bootstrap=True,           # Enable bagging for regularization
    max_leaf_nodes=500,       # Significant reduction to limit model complexity
    max_samples=0.8,          # Subsample training data for each tree
    random_state=42,
    n_jobs=-1,
)

# XGBoost and LightGBM parameters removed to simplify dependencies
# Random Forest provides the best performance for this dataset

# ----------------------------
# Visualization configuration
# ----------------------------

# Manual color palette for cruise visualizations
# Colors will be assigned sequentially to cruises in sorted order
# If there are more cruises than manual colors, remaining cruises use automatic tab20 colors
CRUISE_MANUAL_COLORS = [
    "#D73027",  # red
    "#FC8D59",  # orange
    "#FEE08B",  # light yellow
    "#D9EF8B",  # lime
    "#91CF60",  # green
    "#1A9850",  # deep green
    "#66C2A5",  # teal
    "#3288BD",  # cyan/blue
    "#5E4FA2",  # indigo
    "#9E0142",  # magenta
    "#F46D43",  # warm orange
    "#FDAE61",  # peach
    "#ABD9E9",  # pale cyan
    "#B5B5B5",  # light gray
    "#8C510A",  # brown
    "#505050",  # dark gray
    ]

# Set to empty list [] or None to disable manual colors entirely by uncommenting:
# CRUISE_MANUAL_COLORS = []

# Optional mapping of cruise identifiers to user-friendly names. Example:
# CRUISE_NAME_MAP = {23: 'PACE Cruise May 2024', 42: 'GPIG Cruise 2025'}
# If None or empty, the plotting code will fall back to default labels like 'Dataset 23'.
#CRUISE_NAME_MAP = None
CRUISE_NAME_MAP = {
    1: 'NAAMES 01',
    2: 'NAAMES 02',
    3: 'NAAMES 03',
    4: 'NAAMES 04',
    5: 'EXPORTS 01',
    6: 'EXPORTS 02',
    7: 'PEACETIME',
    8: 'TaraMM-1',
    9: 'TaraMM-2',
    10: 'TaraMM-3',
    11: 'TaraMM-4',
    12: 'TaraMM-5',
    13: 'TaraMM-6',
    14: 'SO-PACE KM2418',
    15: 'SO-PACE KM2419',
    16: 'SO-PACE TGT444'}