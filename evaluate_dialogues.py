import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime  # Import the datetime module

# --- Configuration ---
PROMPT_FILE_PATH = os.path.join('prompts', 'evaluation_prompt.txt')
OUTPUT_FOLDER = 'output'
SAMPLED_FILES_CSV = os.path.join(OUTPUT_FOLDER, 'sampled_dialogue_paths.csv')
# The final CSV filename is now generated dynamically in main()


def setup_folders(folders):
    """Ensure that the necessary folders exist."""
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def evaluate_dialogue_with_openai(client: OpenAI, dialogue_content: str, prompt_template: str) -> dict:
    """Sends a dialogue to the OpenAI API for evaluation using a structured prompt."""
    full_prompt = prompt_template.format(dialogue_content=dialogue_content)
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a helpful clinical evaluation\
                 assistant designed to output JSON."},
                {"role": "user", "content": full_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  - An error occurred with the OpenAI API: {e}")
        return {"error": str(e)}


def main():
    """Main function to run the evaluation process on pre-sampled files."""
    print("--- Running Script 2: Evaluate Dialogues ---")

    # --- MODIFIED: Generate a unique filename with a timestamp ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_evaluation_csv = os.path.join(OUTPUT_FOLDER, f'dialogue_evaluations_{timestamp}.csv')
    # --- End of modification ---

    # Setup
    load_dotenv()
    # Ensure prompts folder exists since we read from it
    setup_folders([PROMPT_FILE_PATH.split(os.sep)[0], OUTPUT_FOLDER])

    # --- Step 1: Load API Key, Prompt, and Sampled File List ---
    print("Step 1: Initializing and loading data...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("FATAL: OPENAI_API_KEY not found in .env file.")
        return

    try:
        client = OpenAI(api_key=api_key)
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

    # --- ADDED FOR TESTING: Limit to the first 10 dialogues ---
    # To run on all 380 files, comment out or remove the next line
    file_paths_to_process = file_paths_to_process[:10]  # Limit to first 10 for testing
    print("\n!!! --- RUNNING IN TEST MODE --- !!!")
    print(f"Processing only the first {len(file_paths_to_process)} dialogues from the sample list.")
    # --- End of testing block ---

    # --- Step 2: Process and Evaluate Each Dialogue ---
    print("\nStep 2: Evaluating dialogues with OpenAI API...")
    all_results = []
    for file_path in tqdm(file_paths_to_process, desc="Evaluating Dialogues"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dialogue_content = f.read()
        except Exception as e:
            print(f"Warning: Could not read file {os.path.basename(file_path)}. Skipping. Error: {e}")
            continue

        evaluation = evaluate_dialogue_with_openai(client, dialogue_content, prompt_template)

        flat_result = {"File_Name": os.path.basename(file_path), "Dialogue_Content": dialogue_content}
        if "error" in evaluation:
            flat_result["Evaluation_Error"] = evaluation["error"]
        else:
            for key, value in evaluation.items():
                flat_result[f"{key.replace('_', ' ').title()}_Score"] = value.get('score')
                flat_result[f"{key.replace('_', ' ').title()}_Justification"] = value.get('justification')
        all_results.append(flat_result)

    # --- Step 3: Save Results to CSV ---
    if not all_results:
        print("No results to save. Exiting.")
        return

    final_df = pd.DataFrame(all_results)
    # Use the new timestamped filename
    final_df.to_csv(final_evaluation_csv, index=False, encoding='utf-8-sig')
    print(f"\n-> Successfully saved final evaluations to '{os.path.abspath(final_evaluation_csv)}'")

    print("\nEvaluation process complete.")


if __name__ == '__main__':
    main()
