# -*- coding: utf-8 -*-
"""
evaluate_llm_output.py

This script provides a framework for evaluating the consistency and quality
of the structured output from the advanced LLM analysis.

It checks if the LLM's narrative and classifications are consistent with the
input data's risk factors.

Steps:
1.  Loads the original pilot dataset and the JSON report from the advanced LLM analysis.
2.  Merges the two data sources.
3.  For each LLM-analyzed record, it runs a series of consistency checks.
4.  Calculates a "consistency score" for each record.
5.  Saves a final evaluation report highlighting potential inconsistencies.
"""

import pandas as pd
import json

def evaluate_consistency():
    """
    Main function to evaluate the consistency of LLM-generated narratives.
    """
    try:
        with open("advanced_llm_analysis_report.json", 'r', encoding='utf-8') as f:
            llm_results = json.load(f)
        llm_df = pd.DataFrame(llm_results)
        data = pd.read_csv(PILOT_DATASET_CSV)
        print("Successfully loaded source data and LLM analysis report.")
    except FileNotFoundError:
        print("Error: Required files not found. Please run 'generate_data.py' and 'advanced_llm_analysis.py' first.")
        return

    # Merge LLM results with original data for cross-checking
    eval_df = pd.merge(llm_df, data, left_on='original_id', right_on='id', how='left')

    print("\nStarting consistency evaluation of LLM outputs...")
    scores = []
    issues = []

    for index, row in eval_df.iterrows():
        score = 0
        issue_list = []
        
        # Check 1: If original data has off-hour/weekend flags, does the narrative mention it?
        if row.get('is_offhour') or row.get('is_weekend'):
            if "hour" in row['soc_narrative'].lower() or "weekend" in row['soc_narrative'].lower() or "night" in row['soc_narrative'].lower():
                score += 1
            else:
                issue_list.append("Narrative missed off-hour/weekend context.")

        # Check 2: If data exfiltration is the intent, are attachments/size mentioned?
        if row.get('potential_intent') == 'Data Exfiltration':
            if "attachment" in row['soc_narrative'].lower() or "large" in row['soc_narrative'].lower():
                score += 1
            else:
                issue_list.append("Data Exfiltration intent lacks mention of attachments/size.")

        # Check 3: If BCC was used, is the intent related to covert activity?
        if row.get('has_bcc'):
            if "covert" in row.get('potential_intent', '').lower() or "bcc" in row['soc_narrative'].lower():
                score += 1
            else:
                issue_list.append("BCC usage was not reflected in intent or narrative.")
        
        # Simple score normalization (example)
        consistency_score = (score / 3.0) * 100
        scores.append(f"{consistency_score:.2f}%")
        issues.append("; ".join(issue_list) if issue_list else "None")

    eval_df['consistency_score'] = scores
    eval_df['detected_issues'] = issues

    # --- Save Evaluation Report ---
    output_file = "llm_evaluation_report.csv"
    eval_df.to_csv(output_file, index=False)

    print(f"\nEvaluation complete. Report saved to '{output_file}'.")
    print("\n--- Evaluation Report Preview ---")
    print(eval_df[['original_id', 'user', 'potential_intent', 'consistency_score', 'detected_issues']].head())
    print("---------------------------------")

if __name__ == "__main__":
    evaluate_consistency()
