# -*- coding: utf-8 -*-
"""
main.py

The main orchestrator script for the Insider Threat Detection project.
This script runs the entire pipeline in the correct sequence, from data
generation to final analysis and reporting.

Running this single script will execute the following steps:
1.  Generate the labeled pilot dataset (`generate_data.py`).
2.  Train and evaluate baseline models (`train_models.py`).
3.  Analyze and visualize feature importance (`analyze_feature_importance.py`).
4.  Generate key result visualizations (`visualize_results.py`).
5.  Generate SOC-style narratives for high-risk cases (`llm_narrative_generator.py`).
6.  Perform a user-level behavior analysis (`analyze_user_behavior.py`).
"""

import subprocess
import sys

def run_script(script_name):
    """A helper function to run a python script and handle errors."""
    try:
        print(f"\n{'='*20} Running: {script_name} {'='*20}")
        # Use sys.executable to ensure the same python interpreter is used
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("--- Stderr ---")
            print(result.stderr)
        print(f"{'-'*20} Finished: {script_name} {'-'*20}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        # Decide whether to exit or continue
        sys.exit(f"Pipeline stopped due to an error in {script_name}.")
    except FileNotFoundError:
        print(f"Error: The script '{script_name}' was not found. Please ensure all scripts are in the same directory.")
        sys.exit()

def run_pipeline():
    """
    Executes the full project pipeline.
    """
    print("Starting the full insider threat detection pipeline...")

    # Define the sequence of scripts to run
    pipeline_scripts = [
        "generate_data.py",
        "train_models.py",
        "analyze_feature_importance.py",
        "visualize_results.py",
        "llm_narrative_generator.py",
        "analyze_user_behavior.py"
    ]
    
    for script in pipeline_scripts:
        run_script(script)
        
    print("\n=======================================================")
    print("✅ Pipeline completed successfully!")
    print("All datasets, models, reports, and figures have been generated.")
    print("=======================================================")

if __name__ == "__main__":
    run_pipeline()
