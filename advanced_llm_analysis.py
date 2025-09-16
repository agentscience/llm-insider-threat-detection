# -*- coding: utf-8 -*-
"""
advanced_llm_analysis.py

This script represents an advanced implementation of the Stage-2 LLM Semantic Review.
Instead of generating a simple narrative, it uses dynamic prompts to ask the LLM
to perform a multi-faceted analysis, acting as a virtual SOC analyst.

Key functionalities:
1.  Loads high-risk samples from the pilot dataset.
2.  Uses different prompt templates based on the identified risk profile.
3.  Instructs the LLM to return a structured JSON object containing:
    - A concise SOC-style narrative.
    - A classification of the potential intent.
    - Extraction of key entities (sensitive keywords, recipients).
    - A suggested next action for a human analyst.
4.  Saves the structured analysis to a JSON file.
"""

import pandas as pd
import json
from config import PILOT_DATASET_CSV, NUM_NARRATIVE_SAMPLES

def mock_advanced_llm_call(prompt: str) -> str:
    """
    A mock function to simulate an advanced LLM call that returns a JSON string.
    This simulates the model's ability to reason and structure its output.
    """
    intent = "Unknown"
    action = "Review Manually"
    entities = {"keywords": [], "external_recipients": []}
    narrative = "The activity exhibits several risk factors requiring analyst attention."

    if "data exfiltration" in prompt:
        intent = "Data Exfiltration"
        action = "Escalate to IT Security Immediately"
        narrative = "Potential data exfiltration detected. The user sent an email with multiple large attachments outside of business hours to external parties."
        if "client list" in prompt: entities["keywords"].append("client list")
        if "BCC" in prompt: narrative += " The use of BCC suggests an attempt to hide the communication."

    elif "covert communication" in prompt:
        intent = "Covert Communication / Policy Violation"
        action = "Monitor User's Communications"
        narrative = "Suspicious internal communication pattern observed. The email was sent on a weekend to multiple colleagues, deviating from the normal workflow."
        if "contract" in prompt: entities["keywords"].append("contract")

    else:
        intent = "Policy Violation"
        action = "Log and Monitor"
        narrative = "The user's action violates standard company policy by sending emails late at night."

    # Simulate extracting recipients
    if "external address" in prompt:
        entities["external_recipients"].append("example@competitor.com")

    # Return a JSON formatted string
    return json.dumps({
        "soc_narrative": narrative,
        "potential_intent": intent,
        "extracted_entities": entities,
        "suggested_action": action
    }, indent=4)

def generate_advanced_analysis(num_samples: int = NUM_NARRATIVE_SAMPLES):
    """
    Main function to generate advanced, multi-faceted analysis for high-risk emails.
    """
    try:
        data = pd.read_csv(PILOT_DATASET_CSV)
    except FileNotFoundError:
        print(f"Error: '{PILOT_DATASET_CSV}' not found. Please run the data generation script first.")
        return

    high_risk_samples = data[data['label'] == 1].sort_values("total_risk", ascending=False).head(num_samples)
    print(f"\nGenerating advanced LLM analysis for the top {len(high_risk_samples)} suspicious samples...")

    results = []
    for index, row in high_risk_samples.iterrows():
        # --- Dynamic Prompt Construction ---
        prompt_context = f"Analyze the following high-risk email from user '{row['user']}'."
        risk_profile = []
        
        # Profile for potential data exfiltration
        if row.get('is_big_size') and row.get('has_attach') and row.get('is_offhour'):
            prompt_context += " The profile suggests potential data exfiltration."
            risk_profile.append("data exfiltration")
        # Profile for covert communication
        elif row.get('is_weekend') and row.get('num_recipients') > 3:
            prompt_context += " The profile suggests potential covert communication."
            risk_profile.append("covert communication")
        
        content_snippet = str(row.get('content', ''))[:150]
        prompt = f"""
        **Instruction**: Act as a Tier 2 SOC Analyst. Based on the following information, provide a structured JSON analysis.
        
        **Context**: {prompt_context}
        
        **Email Metadata**:
        - Timestamp: {row['date']}
        - Risk Score: {row['total_risk']:.2f}
        - Key Risk Factors: {risk_profile}
        - Content Snippet: "{content_snippet}..."

        **Required JSON Output Format**:
        {{
            "soc_narrative": "A concise summary for a security report.",
            "potential_intent": "Classify into one of: 'Data Exfiltration', 'Policy Violation', 'Covert Communication', 'Phishing/Spam', 'Unknown'.",
            "extracted_entities": {{ "keywords": ["list of sensitive terms found"], "external_recipients": ["list of external email addresses"] }},
            "suggested_action": "Recommend a next step from: 'Escalate to IT Security Immediately', 'Monitor User's Communications', 'Log and Monitor', 'No Action Needed'."
        }}
        """
        
        # --- (Mock) LLM Call ---
        llm_output_str = mock_advanced_llm_call(prompt)
        
        try:
            llm_output_json = json.loads(llm_output_str)
            llm_output_json['original_id'] = row['id']
            llm_output_json['user'] = row['user']
            results.append(llm_output_json)
        except json.JSONDecodeError:
            print(f"Warning: LLM returned invalid JSON for email ID {row['id']}.")

    # --- Save Results ---
    output_file = "advanced_llm_analysis_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"\nSuccessfully generated advanced analysis. Report saved to '{output_file}'.")
    print("\n--- Example of First Analysis Result ---")
    print(json.dumps(results[0], indent=4))
    print("----------------------------------------")

if __name__ == "__main__":
    generate_advanced_analysis()
