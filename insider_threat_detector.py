# -*- coding: utf-8 -*-
"""
Insider Threat Detection from Email Data.

This script loads email data, performs preprocessing and feature engineering,
detects anomalies using Isolation Forest to create labels, and then trains
and evaluates various classification models to identify potential insider threats.
"""

# 1. Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.utils import to_categorical
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.base')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn._oldcore')
warnings.filterwarnings("ignore", category=UserWarning, module='keras.src.layers.rnn.rnn')

# 2. Function Definitions

def load_and_preprocess_data(file_path, sample_frac=0.1):
    """
    Loads data from a CSV file, performs initial cleaning and feature engineering.

    Args:
        file_path (str): The path to the email CSV file.
        sample_frac (float): Fraction of data to sample for processing.

    Returns:
        pd.DataFrame: A preprocessed sample of the original DataFrame.
    """
    print("Loading and preprocessing data...")
    data = pd.read_csv(file_path)
    
    # Take a sample to reduce memory and computation time
    data_sample = data.sample(frac=sample_frac, random_state=42)
    
    # Data cleaning and feature engineering
    data_sample['date'] = pd.to_datetime(data_sample['date'], format='%m/%d/%Y %H:%M:%S')
    data_sample['cc'] = data_sample['cc'].fillna('')
    data_sample['bcc'] = data_sample['bcc'].fillna('')
    data_sample['num_recipients'] = (
        data_sample['to'].str.count(';') + 
        data_sample['cc'].str.count(';') + 
        data_sample['bcc'].str.count(';') + 1
    )
    data_sample['hour'] = data_sample['date'].dt.hour
    data_sample['day_of_week'] = data_sample['date'].dt.dayofweek
    
    print(f"Data loaded. Using a sample of {len(data_sample)} records.")
    return data_sample

def create_features_and_labels(data_sample):
    """
    Vectorizes email content and combines features. Uses IsolationForest to create anomaly labels.

    Args:
        data_sample (pd.DataFrame): The preprocessed DataFrame sample.

    Returns:
        tuple: A tuple containing scaled feature matrix (X_scaled) and labels (y).
    """
    print("Creating features and generating anomaly labels...")
    
    # Vectorize content using TF-IDF
    tfidf = TfidfVectorizer(max_features=1000)
    content_tfidf = tfidf.fit_transform(data_sample['content']).toarray()
    content_tfidf_df = pd.DataFrame(content_tfidf, columns=tfidf.get_feature_names_out())

    # Combine numeric features with TF-IDF features
    numeric_features = ['size', 'attachments', 'num_recipients', 'hour', 'day_of_week']
    X_numeric = data_sample[numeric_features]
    
    # Standardize numeric features
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    X_numeric_scaled_df = pd.DataFrame(X_numeric_scaled, columns=numeric_features)

    # Concatenate all features
    X_scaled = pd.concat([
        X_numeric_scaled_df.reset_index(drop=True),
        content_tfidf_df.reset_index(drop=True)
    ], axis=1)

    # Use IsolationForest to detect anomalies and create the 'anomaly' column
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    anomaly_predictions = iso_forest.fit_predict(X_scaled)
    
    # Convert predictions from {1, -1} to {0, 1} (0: normal, 1: anomaly)
    y = np.array([0 if pred == 1 else 1 for pred in anomaly_predictions])
    
    print("Features and labels created.")
    return X_scaled, y

def train_evaluate_sklearn(model, X_train, y_train, X_test, y_test, model_name):
    """
    Trains and evaluates a scikit-learn model.

    Args:
        model: The scikit-learn model instance.
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        model_name (str): The name of the model for printing results.
    """
    print(f"\n--- Training {model_name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name}: {acc*100:.3f} %")

def train_evaluate_keras_model(model, X_train, y_train, X_test, y_test, model_name, epochs=10, batch_size=64):
    """
    Trains and evaluates a Keras (TensorFlow) model (LSTM/GRU).

    Args:
        model: The Keras model instance.
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        model_name (str): The name of the model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    """
    print(f"\n--- Training {model_name} ---")
    
    # Reshape data for RNNs: (samples, timesteps, features)
    X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train)
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(
        X_train_reshaped, y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=2
    )
    
    y_pred_proba = model.predict(X_test_reshaped)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name}: {acc*100:.3f} %")

# 3. Main Execution Block
def main():
    """
    Main function to run the entire threat detection pipeline.
    """
    # Define file path for the dataset
    FILE_PATH = 'email.csv'
    
    # Step 1: Load and preprocess data
    data_sample = load_and_preprocess_data(FILE_PATH, sample_frac=0.1)
    
    # Step 2: Create features and generate anomaly labels
    X, y = create_features_and_labels(data_sample)
    
    # Step 3: Split the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")
    
    # Step 4: Train and evaluate Scikit-learn models
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_evaluate_sklearn(rf_model, X_train, y_train, X_test, y_test, "Random Forest")
    
    # Gradient Boosting
    gbdt_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    train_evaluate_sklearn(gbdt_model, X_train, y_train, X_test, y_test, "Gradient Boosting")
    
    # Multi-Layer Perceptron (MLP)
    mlp_model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=100, random_state=42)
    train_evaluate_sklearn(mlp_model, X_train, y_train, X_test, y_test, "Multi-Layer Perceptron")
    
    # Stacking Ensemble
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gbdt', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]
    stacking_model = StackingClassifier(
        estimators=estimators, 
        final_estimator=RandomForestClassifier(n_estimators=100, random_state=42)
    )
    train_evaluate_sklearn(stacking_model, X_train, y_train, X_test, y_test, "Stacking Ensemble")

    # Step 5: Train and evaluate Keras models
    input_shape = (1, X_train.shape[1])

    # LSTM Model
    lstm_model = Sequential([
        LSTM(50, input_shape=input_shape, return_sequences=True),
        LSTM(50),
        Dense(2, activation='softmax')
    ])
    train_evaluate_keras_model(lstm_model, X_train, y_train, X_test, y_test, "LSTM")

    # GRU Model
    gru_model = Sequential([
        GRU(50, input_shape=input_shape, return_sequences=True),
        GRU(50),
        Dense(2, activation='softmax')
    ])
    train_evaluate_keras_model(gru_model, X_train, y_train, X_test, y_test, "GRU")
    
    print("\n--- All models trained and evaluated. ---")

if __name__ == "__main__":
    main()
