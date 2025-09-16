# -*- coding: utf-8 -*-
"""
predict_risk.py

This script demonstrates how to use a trained model for real-time inference.
It performs the following steps:
1.  Loads a pre-trained model from disk (e.g., the Stacking Ensemble).
2.  Takes a new, unseen piece of data (as a dictionary).
3.  Preprocesses this data in the same way as the training data.
4.  Predicts whether the activity is 'Normal' or 'Suspicious'.
5.  Outputs the prediction and the associated probability.

This simulates how the model would be deployed to score new events in a live environment.
"""

import pandas as pd
import joblib
from config import TRAINING_FEATURES, SAVED_MODEL_PATH, RANDOM_STATE, TEST_SET_SIZE
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_and_save_model_if_not_exists(data_path="small_insider_pilot.csv"):
    """
    A helper function to train and save a model if one doesn't already exist.
    This ensures the prediction script can be run independently.
    """
    try:
        joblib.load(SAVED_MODEL_PATH)
        print(f"Loaded existing model from '{SAVED_MODEL_PATH}'.")
        return
    except FileNotFoundError:
        print(f"No existing model found. Training and saving a new model to '{SAVED_MODEL_PATH}'.")
        data = pd.read_csv(data_path)
        X = data[TRAINING_FEATURES].fillna(0)
        y = data['label']

        X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE)
        
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)

        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
            ('lr', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
        ]
        model = StackingClassifier(estimators=estimators, final_estimator=MLPClassifier(random_state=RANDOM_STATE))
        model.fit(X_train_scaled, y_train)

        # Save both the model and the scaler
        joblib.dump({'model': model, 'scaler': scaler}, SAVED_MODEL_PATH)
        print("Model training complete and saved.")


def predict_single_instance(instance_data):
    """
    Loads a trained model and scaler to predict the risk of a single data instance.
    
    Args:
        instance_data (dict): A dictionary representing a single email event.
    """
    try:
        artifacts = joblib.load(SAVED_MODEL_PATH)
        model = artifacts['model']
        scaler = artifacts['scaler']
    except FileNotFoundError:
        print(f"Error: Model file not found at '{SAVED_MODEL_PATH}'. Please train the model first.")
        return

    # Convert the dictionary to a DataFrame with the correct feature order
    instance_df = pd.DataFrame([instance_data], columns=TRAINING_FEATURES).fillna(0)
    
    # Scale the features using the loaded scaler
    instance_scaled = scaler.transform(instance_df)
    
    # Make prediction
    prediction = model.predict(instance_scaled)[0]
    probability = model.predict_proba(instance_scaled)[0]
    
    # Output results
    status = "Suspicious" if prediction == 1 else "Normal"
    print("\n--- Prediction Result ---")
    print(f"Predicted Status: {status}")
    print(f"Probability [Normal, Suspicious]: [{probability[0]:.4f}, {probability[1]:.4f}]")
    print("-------------------------")


if __name__ == "__main__":
    # Ensure a model exists before trying to predict
    train_and_save_model_if_not_exists()

    # --- Example of a new, unseen email event ---
    # This simulates a potentially risky event
    new_suspicious_email = {
        'size': 60000,          # Large email size
        'attachments': 5,       # Multiple attachments
        'num_recipients': 3,
        'hour': 2,              # Off-hour (2 AM)
        'day_of_week': 6,       # Weekend (Sunday)
        'is_offhour': 1,
        'is_weekend': 1,
        'has_attach': 1,
        'has_bcc': 1,           # BCC was used
        'is_big_size': 1,
        'is_many_recip': 0,
        'has_sensitive_kw': 1,  # Contains sensitive keywords
        'user_psy_risk': 1.5    # User has a high psychometric risk score
    }
    
    print("\nPredicting risk for a new suspicious email instance...")
    predict_single_instance(new_suspicious_email)
    
    # Example of a clearly normal event
    new_normal_email = {
        'size': 1500,
        'attachments': 0,
        'num_recipients': 1,
        'hour': 11,             # Normal working hours
        'day_of_week': 1,       # Weekday (Tuesday)
        'is_offhour': 0,
        'is_weekend': 0,
        'has_attach': 0,
        'has_bcc': 0,
        'is_big_size': 0,
        'is_many_recip': 0,
        'has_sensitive_kw': 0,
        'user_psy_risk': -0.8
    }

    print("\nPredicting risk for a new normal email instance...")
    predict_single_instance(new_normal_email)
