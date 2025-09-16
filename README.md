# LLM-Enhanced Semantic Analysis for Insider Threat Detection

A two-stage pipeline for detecting insider threats in enterprise email communication logs by combining scalable behavioural anomaly filtering with deep semantic analysis from Large Language Models (LLMs).

---

### Research Objectives
This study addresses three central research questions: *Submitted to 1st Open Conference on AI Agents for Science (agents4science 2025).*

1.  **Can a scalable, hybrid risk-scoring model (Stage 1) effectively filter a large volume of communication logs to identify a small, high-risk subset of potentially malicious activities?** 
2.  **How can Large Language Models (LLMs) be applied in a second stage (Stage 2) to provide deep semantic analysis and generate actionable, human-readable SOC-style narratives for these high-risk events?** 
3.  **Does this two-stage pipeline provide a practical and cost-effective framework that improves upon traditional anomaly detection by incorporating semantic context and intent analysis, while aligning with real-world Security Operations Centre (SOC) workflows?** 

### Methodology Overview
Our approach is a two-stage pipeline designed to mirror the workflow of a modern Security Operations Centre (SOC), balancing automated filtering with high-fidelity analysis.

* **Stage 1: Behavioural Anomaly Filtering**
    This stage processes the entire dataset at scale. We perform feature engineering on email metadata (e.g., size, recipients, attachments) and user psychometric profiles. A hybrid risk score is calculated for each email by combining rule-based flags, statistical deviations, and an unsupervised **Isolation Forest** anomaly score. This allows us to efficiently filter millions of events down to a small, manageable subset of high-risk candidates (e.g., the top 1%).

* **Stage 2: LLM Semantic Review**
    The high-risk subset from Stage 1 is passed to this stage for in-depth analysis. We use a Large Language Model (LLM) to act as a virtual SOC analyst. For each high-risk email, a structured prompt containing its metadata and risk factors is sent to the LLM. The model then generates a structured output that includes a concise SOC-style narrative, a classification of potential intent (e.g., Data Exfiltration), and a suggested next action for a human analyst.

### Project Structure
This repository is organized into a modular pipeline, with each script performing a specific task.

```
.
├── data/
│   ├── email.csv             # (Must be obtained from the dataset source)
│   └── psychometric.csv      # (Must be obtained from the dataset source)
├── main.py                   # Main orchestrator to run the entire pipeline
├── config.py                 # Central configuration file for all parameters
│
├── 1_data_generation/
│   └── generate_data.py      # Stage 1: Processes raw data to create the labeled pilot dataset
│
├── 2_modelling/
│   ├── train_models.py       # Trains and evaluates baseline ML models
│   ├── predict_risk.py       # Simulates real-time prediction on a new data instance
│   └── tune_hyperparameters.py # Finds optimal model parameters using GridSearchCV
│
├── 3_analysis/
│   ├── analyze_feature_importance.py # Analyzes and plots which features are most predictive
│   ├── analyze_user_behavior.py    # Generates a report on high-risk and high-volume users
│   └── visualize_results.py        # Creates plots and figures from the paper
│
├── 4_llm_analysis/
│   ├── advanced_llm_analysis.py    # Stage 2: Generates structured analysis using a mock LLM
│   ├── evaluate_llm_output.py      # Evaluates the consistency of the LLM's output
│   └── llm_api_integration_template.py # A template for integrating a real LLM API (e.g., Google Gemini)
│
└── README.md                 # This file
```

### Dataset
This project uses the **CERT Insider Threat Test Dataset**, a collection of synthetic data designed for insider threat research.

* **Source**: Lindauer, Brian (2020). Insider Threat Test Dataset. Carnegie Mellon University. Dataset. [https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247)
* **Description**: The CERT Division generated a collection of synthetic insider threat test datasets that provide both synthetic background data and data from synthetic malicious actors. In this project, we primarily use the `email.csv` and `psychometric.csv` files from release v6.2.
* **Citation**:
    ```
    Lindauer, Brian (2020). Insider Threat Test Dataset. Carnegie Mellon University. Dataset. [https://doi.org/10.1184/R1/12841247.v1](https://doi.org/10.1184/R1/12841247.v1)
    ```

### Setup and Usage

**1. Prerequisites**
* Python 3.8+
* Git

**2. Installation**
```bash
# Clone the repository
git clone [https://github.com/your-username/llm-insider-threat-detection.git](https://github.com/your-username/llm-insider-threat-detection.git)
cd llm-insider-threat-detection

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required libraries
pip install pandas scikit-learn matplotlib seaborn joblib google-generativeai
```

**3. Data Placement**
Download the CERT dataset from the link above and place `email.csv` and `psychometric.csv` into a `data/` directory in the project root.

**4. Running the Full Pipeline**
To run the entire project from data generation to final analysis, simply execute the main script:
```bash
python main.py
```
This will run all the necessary scripts in the correct order and generate all output files (datasets, reports, figures).

**5. Running Individual Scripts**
You can also run individual scripts for specific tasks. For example, to only regenerate the labeled dataset:
```bash
python 1_data_generation/generate_data.py
```

### Citation
If you use this project or methodology in your research, please cite our paper:
```bibtex
@article{insiderthreat2025,
  title   = {LLM-Enhanced Semantic Analysis for Insider Threat Detection in Enterprise Communication Logs},
  author  = {Anonymous, Author},
  journal = {Submitted to 1st Open Conference on AI Agents for Science (agents4science 2025)},
  year    = {2025}
}
```

### Acknowledgments
We thank the CERT Insider Threat Center at the Software Engineering Institute (SEI), Carnegie Mellon University, for creating and making the Insider Threat Test Dataset publicly available.
