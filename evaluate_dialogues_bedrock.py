import os
import json
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime

# Import LangChain components
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage

# --- Configuration ---
PROMPT_FILE_PATH = os.path.join('prompts', 'evaluation_prompt.txt')
OUTPUT_FOLDER = 'output'
SAMPLED_FILES_CSV = os.path.join(OUTPUT_FOLDER, 'sampled_dialogue_paths.csv')

# --- Bedrock Configuration ---
BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def setup_folders(folders):
    """Ensure that the necessary folders exist."""
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def extract_json_from_string(text: str) -> dict | None:
    """
    Finds and parses the first valid JSON object within a string.
    Handles cases where the JSON is embedded in other text and cleans it.
    """
    if not text:
        return None

    # Find the first '{' and the last '}'
    start_index = text.find('{')
    end_index = text.rfind('}')

    if start_index == -1 or end_index == -1 or end_index < start_index:
        return None

    json_str = text[start_index: end_index + 1]

    # --- ADDED: Clean the string of common invalid characters ---
    # The error log shows non-standard whitespace (U+00A0) which json.loads fails on.
    # We replace it with a standard space.
    cleaned_json_str = json_str.replace('\u00a0', ' ')
    # --- End of addition ---

    try:
        # Parse the cleaned string
        return json.loads(cleaned_json_str)
    except json.JSONDecodeError:
        # If it still fails, return None
        return None


def evaluate_dialogue_with_bedrock(llm_client, dialogue_content: str, prompt_template: str) -> dict:
    """Sends a dialogue to Amazon Bedrock for evaluation using LangChain."""

    full_prompt = prompt_template.format(dialogue_content=dialogue_content)

    messages = [
        HumanMessage(content=full_prompt)
    ]

    try:
        response = llm_client.invoke(messages)
        raw_content = response.content

        # Use the robust JSON extractor
        parsed_json = extract_json_from_string(raw_content)

        if parsed_json:
            return parsed_json
        else:
            print(
                f"\n--- FAILED TO PARSE JSON ---\nRaw API Response: {raw_content}\n---------------------------\n")
            return {"error": "Failed to extract valid JSON from API response."}

    except Exception as e:
        print(f"  - An error occurred with the Bedrock API: {e}")
        return {"error": str(e)}


def main():
    """Main function to run the evaluation process on pre-sampled files using Bedrock."""
    print("--- Running Script 2: Evaluate Dialogues (AWS Bedrock with LangChain) ---")

    load_dotenv()
    bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    if bearer_token:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = bearer_token

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_evaluation_csv = os.path.join(OUTPUT_FOLDER, f'dialogue_evaluations_bedrock_{timestamp}.csv')

    setup_folders([PROMPT_FILE_PATH.split(os.sep)[0], OUTPUT_FOLDER])

    print("Step 1: Initializing and loading data...")
    if not bearer_token:
        print("FATAL: AWS_BEARER_TOKEN_BEDROCK not found in .env file.")
        return

    try:
        llm = ChatBedrockConverse(
            model=BEDROCK_MODEL_ID,
            region_name=AWS_REGION,
            temperature=0.1,
            max_tokens=4096
        )

        with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        sampled_df = pd.read_csv(SAMPLED_FILES_CSV)
        file_paths_to_process = sampled_df['file_path'].tolist()

    except FileNotFoundError:
        print(f"FATAL: Could not find '{SAMPLED_FILES_CSV}'.")
        print("Please run '1_sample_dialogues.py' first to generate the sample file list.")
        return
    except Exception as e:
        print(f"FATAL: An initialization error occurred: {e}")
        return

    # --- FOR TESTING: To limit the number of dialogues, uncomment the next line ---
    file_paths_to_process = file_paths_to_process[:10]

    print(f"\nStep 2: Evaluating {len(file_paths_to_process)} dialogues with Bedrock model {BEDROCK_MODEL_ID}...")
    all_results = []
    for file_path in tqdm(file_paths_to_process, desc="Evaluating Dialogues"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dialogue_content = f.read()
        except Exception as e:
            print(f"Warning: Could not read file {os.path.basename(file_path)}. Skipping. Error: {e}")
            continue

        evaluation = evaluate_dialogue_with_bedrock(llm, dialogue_content, prompt_template)

        flat_result = {"File_Name": os.path.basename(file_path), "Dialogue_Content": dialogue_content}
        if "error" in evaluation:
            flat_result["Evaluation_Error"] = evaluation["error"]
        else:
            for key, value in evaluation.items():
                flat_result[f"{key.replace('_', ' ').title()}_Score"] = value.get('score')
                flat_result[f"{key.replace('_', ' ').title()}_Justification"] = value.get('justification')
        all_results.append(flat_result)

    if not all_results:
        print("No results to save. Exiting.")
        return

    final_df = pd.DataFrame(all_results)
    final_df.to_csv(final_evaluation_csv, index=False, encoding='utf-8-sig')
    print(f"\n-> Successfully saved final evaluations to '{os.path.abspath(final_evaluation_csv)}'")

    print("\nEvaluation process complete.")


if __name__ == '__main__':
    main()
