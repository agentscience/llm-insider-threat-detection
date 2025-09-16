# -*- coding: utf-8 -*-
"""
visualize_results.py

This script generates key visualizations based on the processed pilot dataset,
similar to the figures presented in the research paper.

It performs the following steps:
1.  Loads the 'small_insider_pilot.csv' dataset.
2.  Generates boxplots to show feature separation between normal and suspicious labels.
3.  Generates histograms to compare feature distributions.
4.  Trains a Random Forest model to plot a Precision-Recall curve, evaluating
    the performance of the Stage-1 risk scoring.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc

def generate_visualizations():
    """
    Main function to generate and save all visualizations.
    """
    # --- 1. Load Data ---
    try:
        data = pd.read_csv("small_insider_pilot.csv")
        print("Successfully loaded 'small_insider_pilot.csv'.")
    except FileNotFoundError:
        print("Error: 'small_insider_pilot.csv' not found.")
        print("Please run the data generation script first.")
        return

    # For display purposes, let's create a more descriptive label column.
    data['status'] = data['label'].apply(lambda x: 'Suspicious' if x == 1 else 'Normal')

    # --- 2. Generate Boxplots for Feature Separation (like Figure 1a) ---
    print("Generating feature separation boxplots...")
    feature_boxplots = ['size', 'attachments', 'num_recipients', 'hour']
    
    plt.figure(figsize=(20, 5))
    for i, feature in enumerate(feature_boxplots, 1):
        plt.subplot(1, 4, i)
        sns.boxplot(x='status', y=feature, data=data, order=['Normal', 'Suspicious'])
        plt.title(f'{feature.capitalize()} by Label')
        plt.xlabel('Label')
        plt.ylabel(feature.capitalize())
    
    plt.tight_layout()
    plt.savefig("figure_1a_feature_boxplots.png", dpi=300)
    print("Saved: figure_1a_feature_boxplots.png")
    plt.close()

    # --- 3. Generate Histograms for Feature Distribution (like Figure 1b) ---
    print("Generating feature distribution histograms...")
    plt.figure(figsize=(20, 5))
    for i, feature in enumerate(feature_boxplots, 1):
        plt.subplot(1, 4, i)
        sns.histplot(data=data, x=feature, hue='status', element='step', stat='density', common_norm=False)
        plt.title(f'Distribution of {feature.capitalize()} by Label')
    
    plt.tight_layout()
    plt.savefig("figure_1b_feature_histograms.png", dpi=300)
    print("Saved: figure_1b_feature_histograms.png")
    plt.close()

    # --- 4. Generate Precision-Recall Curve (like Figure 2b) ---
    print("Generating Precision-Recall curve...")
    features = [f for f in ['size', 'attachments', 'num_recipients', 'hour', 'day_of_week',
                             'is_offhour', 'is_weekend', 'has_attach', 'has_bcc', 'is_big_size',
                             'is_many_recip', 'has_sensitive_kw', 'user_psy_risk'] if f in data.columns]
    
    X = data[features].fillna(0)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_scores = model.predict_proba(X_test_scaled)[:, 1]
    
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Stage-1 Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig("figure_2b_precision_recall_curve.png", dpi=300)
    print("Saved: figure_2b_precision_recall_curve.png")
    plt.close()
    
    print("\n--- Visualization generation complete. ---")

if __name__ == "__main__":
    generate_visualizations()
