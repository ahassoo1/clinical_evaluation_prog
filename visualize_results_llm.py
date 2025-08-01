import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import matplotlib


# --- Configuration ---
OUTPUT_FOLDER = 'output'
# This file name is for the *original* criteria
VISUALIZATION_FILE_PATH = os.path.join(OUTPUT_FOLDER, 'evaluation_summary_original_criteria.png')


def find_latest_evaluation_file(folder: str) -> str | None:
    """Finds the most recently created evaluation CSV file for the original criteria."""
    search_path = os.path.join(folder, 'dialogue_evaluations_*.csv')
    all_files = glob.glob(search_path)
    # Filter out files related to the new criteria to only visualize the original set
    original_files = [f for f in all_files if 'new_criteria' not in os.path.basename(f)]

    if not original_files:
        return None
    latest_file = max(original_files, key=os.path.getctime)
    return latest_file


def get_viridis_palette(n_colors=3):
    from matplotlib.cm import get_cmap
    cmap = get_cmap('viridis', n_colors)
    # Convert colormap to hex colors
    return [matplotlib.colors.rgb2hex(cmap(i)) for i in range(n_colors)]


def visualize_results(df: pd.DataFrame, output_path: str):
    """Creates and saves a dashboard of visualizations from the original LLM evaluation results."""
    print("Generating visualizations for original LLM criteria...")

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    fig.suptitle("Original Criteria LLM Evaluation Summary", fontsize=24, weight='bold')

    # Correctly identify ONLY the LLM score columns
    llm_score_cols = [col for col in df.columns if col.endswith('_Score') and 'Human' not in col]

    plot_df = df.copy()
    for col in llm_score_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    plot_df.dropna(subset=llm_score_cols, inplace=True)

    if plot_df.empty:
        print("-> No valid data to visualize. Check the evaluation CSV for errors.")
        return

    total_dialogues = len(plot_df)

    # --- Visualization 1: Score Distributions (LLM only) ---
    plot_order = [
        'Factuality_Score', 'Omission_Score', 'Safety_Score',
        'Communication Quality_Score', 'Professionalism Ethics_Score', 'Cultural Sensitivity_Score',
        'Synthetic Dialogue Detection_Score'
    ]

    # Use Viridis palette for the three possible score values (1, 3, 5)
    viridis_palette = get_viridis_palette(3)
    score_palette = {1: viridis_palette[0], 3: viridis_palette[1], 5: viridis_palette[2]}

    # Define specific labels for each criterion's scores
    score_labels_map = {
        'Omission_Score': ['Major Risk', 'Potential Risk', 'No Sig. Risk'],
        'default': ['Poor', 'Adequate', 'Excellent']
    }

    axes_flat = axes.flatten()
    for i, col in enumerate(plot_order):
        if col in plot_df.columns:
            ax = axes_flat[i]
            sns.countplot(x=col, data=plot_df, ax=ax, palette=score_palette, order=[1, 3, 5], hue=col, legend=False)

            clean_title = col.replace('_Score', '').replace('_', ' ')
            ax.set_title(f"LLM Scores: {clean_title}", fontsize=14, weight='bold')

            # Apply the custom labels to the x-axis
            labels = score_labels_map.get(col, score_labels_map['default'])
            ax.set_xticklabels(labels, rotation=0)
            ax.set_xlabel("")  # Remove the generic "Score" label

            ax.set_ylabel("Count", fontsize=10)

            for p in ax.patches:
                height = int(p.get_height())
                percentage = f'({100 * height / total_dialogues:.1f}%)' if total_dialogues > 0 else ''
                label = f'{height}\n{percentage}'
                ax.annotate(label, (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 8), textcoords='offset points',
                            size=9, weight='bold')

    # --- Visualization 2: LLM Correlation Matrix ---
    ax_heatmap = axes_flat[-1]  # Use the last available subplot

    # This ensures the specific column is removed before calculating correlation.
    llm_heatmap_cols = [col for col in llm_score_cols if 'Synthetic Dialogue Detection' not in col]

    llm_corr_df = plot_df[llm_heatmap_cols]
    llm_corr_df.columns = [c.replace('_Score', '').replace('_', ' ') for c in llm_corr_df.columns]

    # Use Spearman's rank correlation for ordinal data
    spearman_corr = llm_corr_df.corr(method='spearman')

    # Changed colormap from 'Reds' to 'viridis'
    sns.heatmap(spearman_corr, annot=True, cmap='viridis_r', fmt=".2f", ax=ax_heatmap, annot_kws={"size": 10})

    ax_heatmap.set_title("LLM Core Criteria Correlation (Spearman's Rank)", fontsize=14, weight='bold')

    plt.setp(ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax_heatmap.get_yticklabels(), rotation=0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"-> Visualizations saved to '{os.path.abspath(output_path)}'")


def main():
    """Main function to load the latest original criteria evaluation data and generate visualizations."""
    print("--- Running Script 3: Visualize Original Criteria Results ---")

    latest_eval_file = find_latest_evaluation_file(OUTPUT_FOLDER)

    if not latest_eval_file:
        print(f"FATAL: No original criteria evaluation CSV files found in the '{OUTPUT_FOLDER}' directory.")
        print("Please run one of the evaluation scripts first.")
        return

    print(f"Loading data from the most recent evaluation file: {os.path.basename(latest_eval_file)}")

    try:
        df = pd.read_csv(latest_eval_file)
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    visualize_results(df, VISUALIZATION_FILE_PATH)
    print("\nVisualization process complete.")


if __name__ == '__main__':
    main()
