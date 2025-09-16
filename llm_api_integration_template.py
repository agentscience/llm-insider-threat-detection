# -*- coding: utf-8 -*-
"""
llm_api_integration_template.py

A template script demonstrating how to integrate a real Large Language Model API
(e.g., Google's Gemini API) into the analysis pipeline.

To use this script:
1.  Install the Google AI library: `pip install -q -U google-generativeai`
2.  Obtain an API key from Google AI Studio.
3.  Set the API key as an environment variable or directly in the script.
    (Using environment variables is highly recommended for security).
"""

import os
import google.generativeai as genai
import json

# --- Configuration ---
# IMPORTANT: It is highly recommended to set your API key as an environment variable
# for security purposes.
# Example in your terminal: export GOOGLE_API_KEY="YOUR_API_KEY"
try:
    # Attempt to get the API key from environment variables
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Fallback for demonstration purposes - replace with your key
        api_key = "YOUR_API_KEY_HERE" 
        print("Warning: API key not found in environment variables. Using placeholder from script.")
    
    genai.configure(api_key=api_key)
    print("Google Gemini API configured successfully.")

except Exception as e:
    print(f"Error configuring the API: {e}")
    print("Please ensure you have a valid API key and the library is installed.")
    api_key = None

def call_gemini_api(prompt: str) -> str:
    """
    Calls the Google Gemini Pro model with a given prompt and returns the text response.
    Includes basic error handling.
    """
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        return json.dumps({
            "soc_narrative": "API key not configured. This is a placeholder response.",
            "potential_intent": "Unknown",
            "extracted_entities": {},
            "suggested_action": "Configure API Key"
        })

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Clean up the response to get a clean JSON string
        # LLMs sometimes wrap JSON in markdown backticks
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        # Validate that the response is valid JSON
        json.loads(cleaned_response) # This will raise an error if invalid
        return cleaned_response

    except Exception as e:
        print(f"An error occurred during the API call: {e}")
        return json.dumps({
            "soc_narrative": f"Error during API call: {e}",
            "potential_intent": "Error",
            "extracted_entities": {},
            "suggested_action": "Check API connection and prompt format."
        })


if __name__ == "__main__":
    # This is an example of how you would use the real API call.
    # This prompt is the same format as in `advanced_llm_analysis.py`.
    example_prompt = """
    **Instruction**: Act as a Tier 2 SOC Analyst. Based on the following information, provide a structured JSON analysis.
    
    **Context**: Analyze the following high-risk email from user 'CDE1234'. The profile suggests potential data exfiltration.
    
    **Email Metadata**:
    - Timestamp: 2023-10-27 02:15:00
    - Risk Score: 9.85
    - Key Risk Factors: ['data exfiltration', 'off-hour', 'BCC']
    - Content Snippet: "Hi, here is the confidential client list you requested for our new venture..."

    **Required JSON Output Format**:
    {{
        "soc_narrative": "A concise summary for a security report.",
        "potential_intent": "Classify into one of: 'Data Exfiltration', 'Policy Violation', 'Covert Communication', 'Phishing/Spam', 'Unknown'.",
        "extracted_entities": {{ "keywords": ["list of sensitive terms found"], "external_recipients": ["list of external email addresses"] }},
        "suggested_action": "Recommend a next step from: 'Escalate to IT Security Immediately', 'Monitor User's Communications', 'Log and Monitor', 'No Action Needed'."
    }}
    """

    print("\n--- Calling Real LLM API with Example Prompt ---")
    api_response_str = call_gemini_api(example_prompt)
    
    print("\n--- Raw API Response ---")
    print(api_response_str)

    try:
        api_response_json = json.loads(api_response_str)
        print("\n--- Parsed JSON Response ---")
        print(json.dumps(api_response_json, indent=4))
    except json.JSONDecodeError:
        print("\nError: The API response was not valid JSON.")
