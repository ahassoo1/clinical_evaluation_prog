# Clinical Dialogue Evaluation Pipeline

This project provides a suite of tools to evaluate the quality of clinical dialogues using Large Language Models (LLMs) like OpenAI's GPT-4 and AWS Bedrock. It automates the process of scoring doctor-patient clinical dialogues based on various clinical and communication metrics, and it generates visualizations to summarize the results.

## Features

- **Automated LLM Evaluation:** Scripts to evaluate dialogue files using OpenAI and AWS Bedrock APIs.
- **Customizable Prompts:** Easily modify the evaluation criteria by editing the prompt template.
- **Data Visualization:** Generates a dashboard of plots summarizing the evaluation scores and their correlations.
- **Reproducible Sampling:** A script to create a consistent sample of dialogues for evaluation.

## Workflow

The evaluation process is designed to be run in a specific order:

1.  **Sample Dialogues:** A subset of dialogues is sampled from the `dialogues/` directory for processing.
2.  **Evaluate Dialogues:** The sampled dialogues are sent to an LLM for evaluation based on a predefined prompt. The results are saved to a CSV file.
3.  **Visualize Results:** The evaluation data is used to generate a summary visualization dashboard.

## Prerequisites

- Python 3.x
- An OpenAI API key or AWS credentials with access to Bedrock.

## Setup and Configuration

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd clinical_integ_eval
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    Create a file named `.env` in the root directory and add your API keys.

    For OpenAI:
    ```
    OPENAI_API_KEY="your-openai-api-key"
    ```

    For AWS Bedrock, ensure your environment is configured with the necessary credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN`).

## Usage

Follow these steps to run the full evaluation pipeline.

### Step 1: Sample the Dialogues

Run the `sample_dialogues.py` script to select a random sample of dialogues from the `dialogues/` folder. This will create a `sampled_dialogue_paths.csv` file in the `output/` directory.

```bash
python sample_dialogues.py
```

### Step 2: Evaluate the Dialogues

Run the evaluation script to process the sampled dialogues. The script will read the file paths from `sampled_dialogue_paths.csv`, send each dialogue to the API for evaluation, and save the results in a new timestamped CSV file in the `output/` folder (e.g., `dialogue_evaluations_YYYYMMDD_HHMMSS.csv`).

**For OpenAI:**
```bash
python evaluate_dialogues.py
```

**For AWS Bedrock:**
```bash
python evaluate_dialogues_bedrock.py
```

### Step 3: Visualize the Results

After the evaluation is complete, run the visualization script. It will automatically find the most recent evaluation file in the `output/` directory and generate a summary image named `evaluation_summary_original_criteria.png`.

```bash
python visualize_results_llm.py
```

## Directory Structure

```
.
├── dialogues/            # Contains the clinical dialogue text files.
├── output/               # Stores all generated files (samples, results, visualizations).
├── prompts/              # Contains the evaluation prompt templates.
├── .env                  # Local environment variables (API keys).
├── requirements.txt      # Python package dependencies.
├── sample_dialogues.py   # Script to sample dialogues for evaluation.
├── evaluate_dialogues.py # Script to run evaluation using OpenAI.
└── visualize_results_llm.py # Script to visualize the LLM evaluation results.
```
