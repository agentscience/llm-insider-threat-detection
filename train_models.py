# -*- coding: utf-8 -*-
"""
train_models.py

This script loads the pre-processed pilot dataset and trains several baseline
machine learning models as described in the research paper "LLM-Enhanced
Semantic Analysis for Insider Threat Detection".

It performs the following steps:
1.  Loads the 'small_insider_pilot.csv' dataset.
2.  Selects relevant features for model training.
3.  Splits the data into training and testing sets.
4.  Trains and evaluates the following models:
    - Logistic Regression
    - Random Forest
    - Multi-Layer Perceptron (MLP)
    - Stacking Ensemble
5.  Prints a classification report and ROC-AUC score for each model,
    mirroring the results in the paper's baseline performance table.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score

def train_and_evaluate_models():
    """
    Main function to run the model training and evaluation pipeline.
    """
    # --- 1. Load Data ---
    try:
        data = pd.read_csv("small_insider_pilot.csv")
        print("Successfully loaded 'small_insider_pilot.csv'.")
        print(f"Dataset shape: {data.shape}")
    except FileNotFoundError:
        print("Error: 'small_insider_pilot.csv' not found.")
        print("Please ensure you have run the data generation script first.")
        return

    # --- 2. Feature Selection and Preprocessing ---
    # As described in the paper, we use the engineered features for classification.
    # We exclude identifiers, raw text, date/time, and the risk scores that were used to create the label.
    features = [
        'size', 'attachments', 'num_recipients', 'hour', 'day_of_week',
        'is_offhour', 'is_weekend', 'has_attach', 'has_bcc',
        'is_big_size', 'is_many_recip', 'has_sensitive_kw',
        'user_psy_risk'
    ]
    
    target = 'label'

    # Ensure all selected feature columns exist
    features = [f for f in features if f in data.columns]
    print(f"\nUsing the following {len(features)} features for training:\n{features}")

    X = data[features].fillna(0)
    y = data[target]

    # --- 3. Split Data and Scale Features ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nData split into training ({len(X_train)}) and testing ({len(X_test)}) sets.")

    # --- 4. Define and Train Models ---
    # Define the models as mentioned in the paper's baseline comparison.
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "MLP (2-layer)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    }

    # Stacking Ensemble requires base estimators
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ]
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=MLPClassifier(random_state=42))
    models["Stacking Ensemble"] = stacking_model
    
    print("\n--- Starting Model Training and Evaluation ---")

    # --- 5. Evaluate and Report Results ---
    for name, model in models.items():
        print(f"\n----- {name} -----")
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Print classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Suspicious (1)']))
        
        # Print ROC-AUC score
        if y_proba is not None:
            roc_auc = roc_auc_score(y_test, y_proba)
            print(f"ROC-AUC Score: {roc_auc:.4f}")
        else:
            print("ROC-AUC Score: Not available for this model.")
            
    print("\n--- Model training and evaluation complete. ---")


if __name__ == "__main__":
    train_and_evaluate_models()
