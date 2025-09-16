# -*- coding: utf-8 -*-
"""
tune_hyperparameters.py

This script performs hyperparameter tuning for the RandomForestClassifier
to find the optimal set of parameters that yields the best performance.

It uses GridSearchCV to systematically search through a predefined grid
of hyperparameters and evaluates them using cross-validation.

Steps:
1.  Loads the pilot dataset.
2.  Defines a parameter grid for the RandomForestClassifier.
3.  Uses GridSearchCV to find the best parameters based on the F1-score.
4.  Prints the best parameters and the corresponding performance score.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from config import PILOT_DATASET_CSV, TRAINING_FEATURES, RANDOM_STATE

def tune_random_forest():
    """
    Main function for hyperparameter tuning of the Random Forest model.
    """
    # --- 1. Load Data ---
    try:
        data = pd.read_csv(PILOT_DATASET_CSV)
        print(f"Successfully loaded '{PILOT_DATASET_CSV}'.")
    except FileNotFoundError:
        print(f"Error: '{PILOT_DATASET_CSV}' not found. Please run the data generation script first.")
        return

    # --- 2. Prepare Data ---
    X = data[TRAINING_FEATURES].fillna(0)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nData prepared for tuning.")

    # --- 3. Define Parameter Grid ---
    # A focused grid of parameters to search for the Random Forest model.
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    print("\nStarting hyperparameter tuning with GridSearchCV...")
    print("This may take a few minutes...")

    # --- 4. Perform Grid Search with Cross-Validation ---
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    
    # We use F1-score as it's a good metric for imbalanced or security-related datasets.
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5, # 5-fold cross-validation
        scoring='f1',
        n_jobs=-1, # Use all available CPU cores
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # --- 5. Report Best Parameters ---
    print("\n--- Hyperparameter Tuning Complete ---")
    print("Best Parameters found:")
    print(grid_search.best_params_)
    
    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    test_f1_score = f1_score(y_test, y_pred)
    
    print(f"\nF1-score of the best model on the test set: {test_f1_score:.4f}")
    print("------------------------------------")

if __name__ == "__main__":
    tune_random_forest()
