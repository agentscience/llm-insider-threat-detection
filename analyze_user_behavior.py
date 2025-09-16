# -*- coding: utf-8 -*-
"""
analyze_user_behavior.py

This script performs a user-level analysis of the dataset, identifying
the top users by email volume and by predicted risk. It generates a summary
report that highlights key behavioral metrics for the most notable users,
aligning with the "Top-user coverage" analysis in the paper.

It performs the following steps:
1.  Loads the 'small_insider_pilot.csv' dataset.
2.  Identifies the top 10 users by total email count.
3.  Identifies the top 10 users by the highest average risk score.
4.  For these users, it calculates statistics like off-hour activity rate,
    BCC usage, etc.
5.  Saves the consolidated report to a CSV file.
"""

import pandas as pd
from config import PILOT_DATASET_CSV, USER_ANALYSIS_REPORT_CSV, TOP_K_USERS

def analyze_users():
    """
    Main function to run the user behavior analysis.
    """
    # --- 1. Load Data ---
    try:
        data = pd.read_csv(PILOT_DATASET_CSV)
        print(f"Successfully loaded '{PILOT_DATASET_CSV}'.")
    except FileNotFoundError:
        print(f"Error: '{PILOT_DATASET_CSV}' not found. Please run the data generation script first.")
        return

    # --- 2. Identify Top Users by Volume and Risk ---
    top_volume_users = data['user'].value_counts().nlargest(TOP_K_USERS).index.tolist()
    
    # Calculate average risk per user and find the top N
    user_risk = data.groupby('user')['total_risk'].mean().nlargest(TOP_K_USERS)
    top_risk_users = user_risk.index.tolist()
    
    # Combine the lists to get a unique set of users to analyze
    users_to_analyze = list(set(top_volume_users + top_risk_users))
    print(f"\nAnalyzing a total of {len(users_to_analyze)} notable users (top by volume and/or risk).")

    # --- 3. Calculate Behavioral Statistics for These Users ---
    user_subset = data[data['user'].isin(users_to_analyze)]
    
    # Define aggregations to perform
    aggregations = {
        'id': 'count',
        'total_risk': 'mean',
        'label': 'mean', # This gives the percentage of suspicious emails
        'is_offhour': 'mean',
        'is_weekend': 'mean',
        'has_attach': 'mean',
        'has_bcc': 'mean',
        'has_sensitive_kw': 'mean'
    }
    
    user_report = user_subset.groupby('user').agg(aggregations).rename(columns={
        'id': 'email_count',
        'total_risk': 'avg_risk_score',
        'label': 'suspicious_rate',
        'is_offhour': 'off_hour_rate',
        'is_weekend': 'weekend_rate',
        'has_attach': 'attachment_rate',
        'has_bcc': 'bcc_rate',
        'has_sensitive_kw': 'sensitive_kw_rate'
    }).sort_values('avg_risk_score', ascending=False)

    # --- 4. Save the Report ---
    user_report.to_csv(USER_ANALYSIS_REPORT_CSV)
    print(f"\nUser behavior analysis complete. Report saved to '{USER_ANALYSIS_REPORT_CSV}'.")
    
    print("\n--- Top 5 Users by Average Risk Score ---")
    print(user_report.head())
    print("-----------------------------------------")
    
if __name__ == "__main__":
    analyze_users()
