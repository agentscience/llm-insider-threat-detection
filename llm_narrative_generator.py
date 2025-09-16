# -*- coding: utf-8 -*-
"""
llm_narrative_generator.py

This script implements Stage-2 of the pipeline: LLM-assisted semantic review.
It takes high-risk events identified in Stage-1 and generates concise,
SOC-style narratives that explain the potential threat.

It performs the following steps:
1.  Loads the 'small_insider_pilot.csv' dataset.
2.  Filters for the top N most suspicious samples.
3.  For each sample, it constructs a structured prompt containing key risk factors.
4.  It uses a mock LLM function to generate a human-readable narrative based on the prompt.
5.  Saves the results to a CSV file for review.

Note: This script uses a *mock* LLM call to demonstrate functionality without
requiring an API key. This can be replaced with a real API call to services
like OpenAI, Google AI, etc.
"""

import pandas as pd

def mock_llm_call(prompt: str) -> str:
    """
    A mock function to simulate a call to a Large Language Model.
    It generates a narrative based on keywords found in the prompt.
    """
    narrative = "Suspicious activity detected. "
    if "late-night" in prompt or "off-hour" in prompt:
        narrative += "Unusual activity occurred outside of standard working hours. "
    if "weekend" in prompt:
        narrative += "Activity took place over the weekend, which deviates from the normal work pattern. "
    if "multiple attachments" in prompt or "large size" in prompt:
        narrative += "The email contains multiple or large attachments, suggesting potential data exfiltration. "
    if "external BCC" in prompt or "has BCC" in prompt:
        narrative += "A Blind Carbon Copy (BCC) was used to an external recipient, indicating an attempt at covert communication. "
    if "sensitive keyword" in prompt:
        keyword = prompt.split("sensitive keyword:")[1].split("'")[1]
        narrative += f"The message includes the sensitive term '{keyword}', possibly related to confidential data leakage. "
    if "high impulsiveness" in prompt:
        narrative += "The user has a high impulsiveness score, which may correlate with poor judgment under stress. "
    
    if narrative == "Suspicious activity detected. ":
        return "A combination of risk factors suggests this activity warrants further investigation."
        
    return narrative.strip()


def generate_soc_narratives(num_samples: int = 10):
    """
    Main function to generate SOC narratives for the highest-risk samples.
    """
    # --- 1. Load Data ---
    try:
        data = pd.read_csv("small_insider_pilot.csv")
        print("Successfully loaded 'small_insider_pilot.csv'.")
    except FileNotFoundError:
        print("Error: 'small_insider_pilot.csv' not found.")
        print("Please run the data generation script first.")
        return

    # --- 2. Filter for Top N Suspicious Samples ---
    high_risk_samples = data[data['label'] == 1].sort_values("total_risk", ascending=False).head(num_samples)
    print(f"\nGenerating narratives for the top {len(high_risk_samples)} suspicious samples...")

    narratives = []
    # --- 3. Construct Prompts and Generate Narratives ---
    for index, row in high_risk_samples.iterrows():
        prompt_parts = [f"Analyse the following email for user '{row['user']}' on {row['date']}:"]
        
        # Collect risk factors
        risk_factors = []
        if row.get('is_offhour'): risk_factors.append("off-hour activity")
        if row.get('is_weekend'): risk_factors.append("weekend activity")
        if row.get('has_attach'): risk_factors.append(f"{row.get('attachments', 'multiple')} attachments")
        if row.get('has_bcc'): risk_factors.append("has BCC")
        if row.get('is_big_size'): risk_factors.append("large size")
        if row.get('has_sensitive_kw') and 'content' in row and isinstance(row['content'], str):
            # A simple search for the first sensitive term
            for term in ["client list", "contract", "password", "source code"]:
                if term in row['content'].lower():
                    risk_factors.append(f"sensitive keyword: '{term}'")
                    break

        if risk_factors:
            prompt_parts.append("Risk factors identified: " + ", ".join(risk_factors) + ".")
        
        # Add a snippet of content if available
        if 'content' in row and isinstance(row['content'], str):
            content_snippet = row['content'][:100] + "..."
            prompt_parts.append(f"Content snippet: \"{content_snippet}\"")
        
        prompt = "\n".join(prompt_parts)
        
        # --- 4. (Mock) LLM Call ---
        narrative = mock_llm_call(prompt)
        
        narratives.append({
            'id': row['id'],
            'user': row['user'],
            'total_risk': row['total_risk'],
            'identified_risks': ", ".join(risk_factors),
            'soc_narrative': narrative
        })

    # --- 5. Save Results ---
    narrative_df = pd.DataFrame(narratives)
    output_file = "soc_narratives_report.csv"
    narrative_df.to_csv(output_file, index=False)
    
    print(f"\nSuccessfully generated narratives. Report saved to '{output_file}'.")
    print("\n--- Sample Narratives ---")
    print(narrative_df.head())
    print("\n--- Narrative generation complete. ---")


if __name__ == "__main__":
    generate_soc_narratives()
