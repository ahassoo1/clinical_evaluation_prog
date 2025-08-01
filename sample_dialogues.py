import os
import random
import pandas as pd

# --- Configuration ---
DIALOGUES_FOLDER = 'dialogues'
OUTPUT_FOLDER = 'output'
SAMPLED_FILES_CSV = os.path.join(OUTPUT_FOLDER, 'sampled_dialogue_paths.csv')

# --- Constants ---
DIALOGUES_PER_DOCTOR = 200
NUM_DOCTORS = 100
TOTAL_SAMPLES = 380


def setup_folders(folders):
    """Ensure that the necessary folders exist."""
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def perform_stratified_sampling(dialogues_folder: str) -> list:
    """
    Performs stratified sampling to get 380 files, ensuring
    representation from every doctor persona.
    Returns a list of absolute file paths.
    """
    print("Performing stratified sampling...")
    if not os.path.isdir(dialogues_folder):
        raise FileNotFoundError(f"Error: The specified dialogues folder does not exist: {dialogues_folder}")

    # Group files by doctor ID
    doctor_files = {}
    for filename in os.listdir(dialogues_folder):
        if filename.startswith("dialog_") and filename.endswith(".txt"):
            try:
                doctor_id = filename.split('_')[1]
                if doctor_id not in doctor_files:
                    doctor_files[doctor_id] = []
                # Store the full, absolute path to the file
                doctor_files[doctor_id].append(os.path.abspath(os.path.join(dialogues_folder, filename)))
            except IndexError:
                continue

    if len(doctor_files) < NUM_DOCTORS:
        print(f"Warning: Found {len(doctor_files)} doctor personas, expected {NUM_DOCTORS}.\
               Sampling will proceed with available data.")

    # Stratified sampling logic
    num_doctors = len(doctor_files)
    num_groups_with_4 = TOTAL_SAMPLES % num_doctors if num_doctors > 0 else 0

    doctor_ids = list(doctor_files.keys())
    random.shuffle(doctor_ids)

    sampled_files = []
    # Sample 4 from the first groups
    for i in range(num_groups_with_4):
        doctor_id = doctor_ids[i]
        k = min(4, len(doctor_files[doctor_id]))
        sampled_files.extend(random.sample(doctor_files[doctor_id], k=k))

    # Sample 3 from the remaining groups
    for i in range(num_groups_with_4, num_doctors):
        doctor_id = doctor_ids[i]
        k = min(3, len(doctor_files[doctor_id]))
        sampled_files.extend(random.sample(doctor_files[doctor_id], k=k))

    print(f"-> Successfully sampled {len(sampled_files)} files.")
    return sampled_files


def main():
    """Main function to run the sampling process."""
    print("--- Running Script 1: Sample Dialogues ---")

    # Setup: Create necessary folders
    setup_folders([DIALOGUES_FOLDER, OUTPUT_FOLDER])

    try:
        sampled_file_paths = perform_stratified_sampling(DIALOGUES_FOLDER)
        if not sampled_file_paths:
            print("No files were sampled. Please check the 'dialogues' folder.")
            return

        # Create a DataFrame with one column for the file paths
        df = pd.DataFrame(sampled_file_paths, columns=['file_path'])

        # Save the list of file paths to a new CSV
        df.to_csv(SAMPLED_FILES_CSV, index=False)

        print(f"\nSuccess! Sampled file paths have been saved to:\n{os.path.abspath(SAMPLED_FILES_CSV)}")

    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please create the 'dialogues' folder and place your .txt files inside before running this script.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == '__main__':
    main()
