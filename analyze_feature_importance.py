# -*- coding: utf-8 -*-
"""
analyze_feature_importance.py

This script trains a RandomForest model and analyzes the importance of each
feature in the decision-making process. Understanding which features are
most influential is crucial for model interpretability.

Steps:
1.  Loads the pilot dataset.
2.  Trains a RandomForestClassifier.
3.  Extracts the feature importances from the trained model.
4.  Creates a bar chart to visualize the top N most important features.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from config import PILOT_DATASET_CSV, TRAINING_FEATURES, RANDOM_STATE

def analyze_importance():
    """
    Main function to calculate and visualize feature importance.
    """
    # --- 1. Load Data ---
    try:
        data = pd.read_csv(PILOT_DATASET_CSV)
        print(f"Successfully loaded '{PILOT_DATASET_CSV}'.")
    except FileNotFoundError:
        print(f"Error: '{PILOT_DATASET_CSV}' not found. Please run the data generation script first.")
        return

    # --- 2. Prepare Data and Train Model ---
    X = data[TRAINING_FEATURES].fillna(0)
    y = data['label']

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X, y)
    print("\nModel trained successfully.")

    # --- 3. Extract and Organize Feature Importances ---
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': TRAINING_FEATURES,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print("\n--- Feature Importances ---")
    print(feature_importance_df)
    print("---------------------------")
    
    # --- 4. Visualize the Results ---
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Feature Importance for Insider Threat Detection')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    output_path = "figure_feature_importance.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nFeature importance plot saved to '{output_path}'.")

if __name__ == "__main__":
    analyze_importance()
