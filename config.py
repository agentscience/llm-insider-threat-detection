# -*- coding: utf-8 -*-
"""
config.py

Central configuration file for the Insider Threat Detection project.
This file contains all the key parameters, file paths, and settings
used across the various scripts in the project.
"""

# --- Data File Paths ---
# Raw data input files
EMAIL_CSV = "email.csv"
PSYCHO_CSV = "psychometric.csv"

# Processed data output files
PILOT_DATASET_CSV = "small_insider_pilot.csv"
SOC_NARRATIVES_CSV = "soc_narratives_report.csv"
USER_ANALYSIS_REPORT_CSV = "user_analysis_report.csv"
SAVED_MODEL_PATH = "trained_model.joblib"

# --- Figure Output Paths ---
BOXPLOT_FIGURE = "figure_1a_feature_boxplots.png"
HISTOGRAM_FIGURE = "figure_1b_feature_histograms.png"
PR_CURVE_FIGURE = "figure_2b_precision_recall_curve.png"

# --- Data Generation Parameters (`generate_data.py`) ---
# Target sample size
TARGET_POS_SAMPLES = 200
TARGET_NEG_SAMPLES = 200

# Top user inclusion policy
TOP_K_USERS = 10
TOP_USER_POS_SAMPLES = 6
TOP_USER_NEG_SAMPLES = 4

# Isolation Forest settings
IFOREST_CONTAMINATION = 0.01

# Feature engineering thresholds (quantiles)
SIZE_QUANTILE = 0.95
RECIPIENT_QUANTILE = 0.95

# Definition of "off-hours"
OFF_HOURS_RANGES = (0, 7, 18, 23)  # 00:00-07:00 & 18:00-23:00
WEEKEND_DAYS = {5, 6} # Saturday, Sunday

# --- Model Training Parameters (`train_models.py`) ---
# Features to be used for training the models
# (Excludes identifiers and the risk scores used to create the label)
TRAINING_FEATURES = [
    'size', 'attachments', 'num_recipients', 'hour', 'day_of_week',
    'is_offhour', 'is_weekend', 'has_attach', 'has_bcc',
    'is_big_size', 'is_many_recip', 'has_sensitive_kw',
    'user_psy_risk'
]

# Data split ratio for training and testing
TEST_SET_SIZE = 0.3

# Random state for reproducibility
RANDOM_STATE = 42

# --- LLM Narrative Generation Parameters (`llm_narrative_generator.py`) ---
# Number of top-risk samples to generate narratives for
NUM_NARRATIVE_SAMPLES = 10
